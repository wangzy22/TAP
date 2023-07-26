import os, time, numpy as np, logging
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from torch import distributed as dist
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from openpoints.utils import setup_logger_dist, Wandb
from openpoints.utils import AverageMeter, resume_checkpoint, load_checkpoint, save_checkpoint, \
    cal_model_parm_nums, set_random_seed, cal_model_parm_nums_separate
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps


def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
    # logger
    logger = setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    # tensorboard
    if cfg.rank == 0:
        # tensorboard
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None

    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logger.info(cfg)

    # build model
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size, model_size_encoder, model_size_generator, model_size_decoder = cal_model_parm_nums_separate(model)
    logging.info(model)
    logging.info('Number of total params: %.4f M' % (model_size / 1e6))
    logging.info('Number of encoder params: %.4f M' % (model_size_encoder / 1e6))
    logging.info('Number of generator params: %.4f M' % (model_size_generator / 1e6))
    logging.info('Number of decoder params: %.4f M' % (model_size_decoder / 1e6))

    if profile:
        model.eval()
        B, N, C = 32, 2048, cfg.model.encoder_args.in_channels
        points = torch.randn(B, N, 3).cuda()
        # from thop import profile as thop_profile
        # macs, params = thop_profile(model, inputs=(points, features))
        # macs = macs / 1e6
        # params = params / 1e6
        # logging.info(f'mac: {macs} \nparams: {params}')

        n_runs = 500
        with torch.no_grad():
            for _ in range(50):  # warm up.
                model(points)
            start_time = time.time()
            for _ in range(n_runs):
                model(points)
                torch.cuda.synchronize()
            time_taken = time.time() - start_time
        print(f'inference time: {time_taken / float(n_runs)}')
        return False

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           split='val',
                                           datatransforms_cfg=cfg.datatransforms,
                                           distributed=cfg.distributed,
                                           )
    logger.info(f"length of training dataset: {len(train_loader.dataset)}")
    logger.info(f"length of validation dataset: {len(val_loader.dataset)}")

    # resume pretrained path
    best_val = np.inf
    if cfg.mode == 'resume':
        cfg.start_epoch, best_val = resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
    elif cfg.mode == 'test':
        resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        logger.info(f"Test mode")

        validate(model, val_loader, cfg)
        validate(model, train_loader, cfg)
        return
    else:
        logging.info('Training from scratch')

    # ===> start training
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
            if hasattr(train_loader.dataset, 'epoch'):
                train_loader.dataset.epoch = epoch - 1

        train_loss, train_vis = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        is_best = False
        if epoch % cfg.val_freq == 0 or epoch == cfg.epochs:
            # Validate the current model
            val_loss, val_vis = validate(model, val_loader, cfg)
            if writer is not None:
                writer.add_scalar('val_loss', val_loss, epoch)
                if epoch % cfg.vis_freq == 0 or epoch == cfg.epochs:
                    for key in val_vis.keys():
                        img_show = make_grid(tensor=val_vis[key][:cfg.vis_num], nrow=2)
                        writer.add_image('val/{}'.format(key), img_show, epoch)
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            logger.info(f'Epoch {epoch} val_loss {val_loss:.3f} best_val {best_val:.3f}')
        save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                        additioanl_dict={'best_val': best_val},
                        is_best=is_best
                        )

        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch} LR {lr:.6f} train_loss {train_loss:.3f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)
            if epoch % cfg.vis_freq == 0 or epoch == cfg.epochs:
                for key in train_vis.keys():
                    img_show = make_grid(tensor=train_vis[key][:cfg.vis_num], nrow=2)
                    writer.add_image('train/{}'.format(key), img_show, epoch)

    save_checkpoint(cfg, model, epoch, optimizer, scheduler, post_fix='clean', is_best=False)
    if writer is not None:
        writer.close()


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    npoints = cfg.num_points
    
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        data['pos'] = data['pos'][:, :, :3].contiguous()
        data['x'] = data['x'][:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()
        num_curr_pts = data['pos'].shape[1]
        if num_curr_pts != npoints:
            data['pos'] = fps(data['pos'], npoints)

        loss, recon_img = model(data)

        loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        loss_meter.update(loss.item())
        lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] LR {lr:.6f} Loss {loss_meter.val:.3f}")
    vis_dict = {
        'recon_pred': recon_img, 
        'recon_gt': data['imgs'].reshape(-1, 3, recon_img.shape[-2], recon_img.shape[-1])
    }
    return loss_meter.avg, vis_dict


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode

    loss_meter = AverageMeter()
    npoints = cfg.num_points

    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda(non_blocking=True)

        data['pos'] = data['pos'][:, :, :3].contiguous()
        data['x'] = data['x'][:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()

        num_curr_pts = data['pos'].shape[1]
        if num_curr_pts != npoints:
            data['pos'] = fps(data['pos'], npoints)

        loss, recon_img = model(data)
        loss_meter.update(loss.item())
        pbar.set_description(f"Test Loss {loss_meter.val:.3f}")

        # image_path = os.path.splitext(data['image_path'][0].split('/')[-1])[0]
        # for n_view in range(recon_img.size(0)):
        #     vis_pred_img = (recon_img[n_view].permute(1,2,0).cpu().numpy() * 255.).astype(np.uint8)
        #     vis_gt_img = (data['imgs'].flatten(0,1)[n_view].permute(1,2,0).cpu().numpy() * 255.).astype(np.uint8)

        #     cv2.imwrite(f'vis/{image_path}_{n_view}_pred.jpg', vis_pred_img)
        #     cv2.imwrite(f'vis/{image_path}_{n_view}_gt.jpg', vis_gt_img)

    vis_dict = {
        'recon_pred': recon_img, 
        'recon_gt': data['imgs'].reshape(-1, 3, recon_img.shape[-2], recon_img.shape[-1])
    }
    return loss_meter.avg, vis_dict
