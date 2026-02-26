import einops
import numpy as np
import torch
import pytorch_lightning as pl
from typing import Dict
from torchvision.utils import make_grid

from tqdm import tqdm
from yacs.config import CfgNode

from lib.datasets.track_dataset import TrackDatasetEval
from lib.models.modules import MANOTransformerDecoderHead, temporal_attention
from hawor.utils.pylogger import get_pylogger
from hawor.utils.render_openpose import render_openpose
from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
from lib.utils.geometry import perspective_projection
from hawor.utils.rotation import angle_axis_to_rotation_matrix
from torch.utils.data import default_collate

from .backbones import create_backbone
from .mano_wrapper import MANO


log = get_pylogger(__name__)
idx = 0

class HAWOR(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        """
        Setup HAWOR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        self.crop_size = cfg.MODEL.IMAGE_SIZE
        self.seq_len = 16
        self.pose_num = 16
        self.pose_dim = 6 # rot6d representation
        self.box_info_dim = 3

        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        try:
            if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
                whole_state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict']
                backbone_state_dict = {}
                for key in whole_state_dict:
                    if key[:9] == 'backbone.':
                        backbone_state_dict[key[9:]] = whole_state_dict[key]
                self.backbone.load_state_dict(backbone_state_dict)
                print(f'Loaded backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                print('WARNING: init backbone from scratch !!!')
        except:
            print('WARNING: init backbone from scratch !!!')

        # Space-time memory
        if cfg.MODEL.ST_MODULE: 
            hdim = cfg.MODEL.ST_HDIM
            nlayer = cfg.MODEL.ST_NLAYER
            self.st_module = temporal_attention(in_dim=1280+3, 
                                                out_dim=1280,
                                                hdim=hdim,
                                                nlayer=nlayer,
                                                residual=True)
            print(f'Using Temporal Attention space-time: {nlayer} layers {hdim} dim.')
        else:
            self.st_module = None

        # Motion memory
        if cfg.MODEL.MOTION_MODULE:
            hdim = cfg.MODEL.MOTION_HDIM
            nlayer = cfg.MODEL.MOTION_NLAYER

            self.motion_module = temporal_attention(in_dim=self.pose_num * self.pose_dim + self.box_info_dim, 
                                                    out_dim=self.pose_num * self.pose_dim,
                                                    hdim=hdim,
                                                    nlayer=nlayer,
                                                    residual=False)
            print(f'Using Temporal Attention motion layer: {nlayer} layers {hdim} dim.')
        else:
            self.motion_module = None

        # Create MANO head
        # self.mano_head = build_mano_head(cfg)
        self.mano_head = MANOTransformerDecoderHead(cfg)

        
        # default open torch compile
        if cfg.MODEL.BACKBONE.get('TORCH_COMPILE', 0): 
            log.info("Model will use torch.compile")
            self.backbone = torch.compile(self.backbone)
            self.mano_head = torch.compile(self.mano_head)

        # Define loss functions
        # self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        # self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        # self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        if cfg.MODEL.get('LOAD_WEIGHTS', None):
            whole_state_dict = torch.load(cfg.MODEL.LOAD_WEIGHTS, map_location='cpu')['state_dict']
            self.load_state_dict(whole_state_dict, strict=True)
            print(f"load {cfg.MODEL.LOAD_WEIGHTS}")

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        if not self.st_module is None:
            all_params += list(self.st_module.parameters())
        if not self.motion_module is None:
            all_params += list(self.motion_module.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        return optimizer

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        image  = batch['img'].flatten(0, 1)
        center = batch['center'].flatten(0, 1)
        scale  = batch['scale'].flatten(0, 1)
        img_focal = batch['img_focal'].flatten(0, 1)
        img_center = batch['img_center'].flatten(0, 1)
        bn = len(image)

        # estimate focal length, and bbox
        bbox_info = self.bbox_est(center, scale, img_focal, img_center)

        # backbone
        feature = self.backbone(image[:,:,:,32:-32])
        feature = feature.float()

        # space-time module
        if self.st_module is not None:
            bb = einops.repeat(bbox_info, 'b c -> b c h w', h=16, w=12)
            feature = torch.cat([feature, bb], dim=1)

            feature = einops.rearrange(feature, '(b t) c h w -> (b h w) t c', t=16)
            feature = self.st_module(feature)
            feature = einops.rearrange(feature, '(b h w) t c -> (b t) c h w', h=16, w=12)

        # smpl_head: transformer + smpl
        # pred_mano_params, pred_cam, pred_mano_params_list = self.mano_head(feature)
        # pred_shape = pred_mano_params_list['pred_shape']
        # pred_pose = pred_mano_params_list['pred_pose']
        pred_pose, pred_shape, pred_cam = self.mano_head(feature)
        pred_rotmat_0 = rot6d_to_rotmat(pred_pose).reshape(-1, self.pose_num, 3, 3)

        # smpl motion module
        if self.motion_module is not None:
            bb = einops.rearrange(bbox_info, '(b t) c -> b t c', t=16)
            pred_pose = einops.rearrange(pred_pose, '(b t) c -> b t c', t=16)
            pred_pose = torch.cat([pred_pose, bb], dim=2)

            pred_pose = self.motion_module(pred_pose)
            pred_pose = einops.rearrange(pred_pose, 'b t c -> (b t) c')

        out = {}
        if 'do_flip' in batch:
            pred_cam[..., 1] *= -1
            center[..., 0] = img_center[..., 0]*2 - center[..., 0] - 1 
        out['pred_cam'] = pred_cam
        out['pred_pose'] = pred_pose
        out['pred_shape'] = pred_shape
        out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(-1, self.pose_num, 3, 3)
        out['pred_rotmat_0'] = pred_rotmat_0
        
        s_out = self.mano.query(out)
        j3d = s_out.joints
        j2d = self.project(j3d, out['pred_cam'], center, scale, img_focal, img_center)
        j2d = j2d / self.crop_size - 0.5 # norm to [-0.5, 0.5]

        trans_full = self.get_trans(out['pred_cam'], center, scale, img_focal, img_center)
        out['trans_full'] = trans_full

        output = {
            'pred_mano_params': {
                'global_orient': out['pred_rotmat'][:, :1].clone(),
                'hand_pose': out['pred_rotmat'][:, 1:].clone(),
                'betas': out['pred_shape'].clone(),
            },
            'pred_keypoints_3d': j3d.clone(),
            'pred_keypoints_2d': j2d.clone(),
            'out': out,
        }
        # print(output)
        # output['gt_project_j2d'] = self.project(batch['gt_j3d_wo_trans'].clone().flatten(0,1), out['pred_cam'], center, scale, img_focal, img_center)
        # output['gt_project_j2d'] = output['gt_project_j2d'] / self.crop_size - 0.5
        

        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['gt_cam_j2d'].flatten(0, 1)
        gt_keypoints_2d = torch.cat([gt_keypoints_2d, torch.ones(*gt_keypoints_2d.shape[:-1], 1, device=gt_keypoints_2d.device)], dim=-1)
        gt_keypoints_3d = batch['gt_j3d_wo_trans'].flatten(0, 1)
        gt_keypoints_3d = torch.cat([gt_keypoints_3d, torch.ones(*gt_keypoints_3d.shape[:-1], 1, device=gt_keypoints_3d.device)], dim=-1)
        pose_gt = batch['gt_cam_full_pose'].flatten(0, 1).reshape(-1, 16, 3)
        rotmat_gt = angle_axis_to_rotation_matrix(pose_gt)
        gt_mano_params = {
            'global_orient': rotmat_gt[:, :1],
            'hand_pose': rotmat_gt[:, 1:],
            'betas': batch['gt_cam_betas'],
        }

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # to avoid nan
        loss_keypoints_2d = torch.nan_to_num(loss_keypoints_2d)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1))

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach() * self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'],
                      loss_keypoints_3d=loss_keypoints_3d.detach() * self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'])

        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach() * self.cfg.LOSS_WEIGHTS[k.upper()]

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True, render_log: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = output['pred_keypoints_2d'].shape[0]
        images = batch['img'].flatten(0,1)
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        losses = output['losses']
        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        
        if render_log:
            gt_keypoints_2d = batch['gt_cam_j2d'].flatten(0,1).clone()
            pred_keypoints_2d = output['pred_keypoints_2d'].clone().detach().reshape(batch_size, -1, 2)
            gt_project_j2d = pred_keypoints_2d
            # gt_project_j2d = output['gt_project_j2d'].clone().detach().reshape(batch_size, -1, 2)

            num_images = 4
            skip=16

            predictions = self.visualize_tensorboard(images[:num_images*skip:skip].cpu().numpy(),
                                                pred_keypoints_2d[:num_images*skip:skip].cpu().numpy(),
                                                gt_project_j2d[:num_images*skip:skip].cpu().numpy(),
                                                gt_keypoints_2d[:num_images*skip:skip].cpu().numpy(),
                                                )
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)
    

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        # pred_mano_params = output['pred_mano_params']
        loss = self.compute_loss(batch, output, train=True)
        
        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        optimizer.step()
        
        # if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
        if self.global_step > 0 and self.global_step % 100 == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True, render_log=self.cfg.TRAIN.get("RENDER_LOG", True))

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=batch_size)

        return output

    def inference(self, imgfiles, boxes, img_focal, img_center, device='cuda', do_flip=False):
        db = TrackDatasetEval(imgfiles, boxes, img_focal=img_focal, 
                        img_center=img_center, normalization=True, dilate=1.2, do_flip=do_flip)

        # Results
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []

        # To-do: efficient implementation with batch
        items = []
        for i in tqdm(range(len(db))):
            item = db[i]
            items.append(item)

            # padding to 16
            if i == len(db) - 1 and len(db) % 16 != 0:
                pad = 16 - len(db) % 16
                for _ in range(pad):
                    items.append(item)

            if len(items) < 16:
                continue
            elif len(items) == 16:
                batch = default_collate(items)
                items = []
            else:
                raise NotImplementedError

            with torch.no_grad():
                batch = {k: v.to(device).unsqueeze(0) for k, v in batch.items() if type(v)==torch.Tensor}
                # for image_i in range(16):
                #     hawor_input_cv2 = vis_tensor_cv2(batch['img'][:, image_i])
                #     cv2.imwrite(f'debug_vis_model.png', hawor_input_cv2)
                #     print("vis")
                output = self.forward(batch)
                out = output['out']

            if i == len(db) - 1 and len(db) % 16 != 0:
                out = {k:v[:len(db) % 16] for k,v in out.items()}
            else:
                out = {k:v for k,v in out.items()}
                
            pred_cam.append(out['pred_cam'].cpu())
            pred_pose.append(out['pred_pose'].cpu())
            pred_shape.append(out['pred_shape'].cpu())
            pred_rotmat.append(out['pred_rotmat'].cpu())
            pred_trans.append(out['trans_full'].cpu())


        results = {'pred_cam': torch.cat(pred_cam),
                'pred_pose': torch.cat(pred_pose),
                'pred_shape': torch.cat(pred_shape),
                'pred_rotmat': torch.cat(pred_rotmat),
                'pred_trans': torch.cat(pred_trans),
                'img_focal': img_focal,
                'img_center': img_center}
        
        return results

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output
    
    def visualize_tensorboard(self, images, pred_keypoints, gt_project_j2d, gt_keypoints):
        pred_keypoints = 256 * (pred_keypoints + 0.5)
        gt_keypoints = 256 * (gt_keypoints + 0.5)
        gt_project_j2d = 256 * (gt_project_j2d + 0.5)
        pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
        gt_keypoints = np.concatenate((gt_keypoints, np.ones_like(gt_keypoints)[:, :, [0]]), axis=-1)
        gt_project_j2d = np.concatenate((gt_project_j2d, np.ones_like(gt_project_j2d)[:, :, [0]]), axis=-1)
        images_np = np.transpose(images, (0,2,3,1))
        rend_imgs = []
        for i in range(images_np.shape[0]):
            pred_keypoints_img = render_openpose(255 * images_np[i].copy(), pred_keypoints[i]) / 255
            gt_project_j2d_img = render_openpose(255 * images_np[i].copy(), gt_project_j2d[i]) / 255
            gt_keypoints_img = render_openpose(255*images_np[i].copy(), gt_keypoints[i]) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2,0,1))
            rend_imgs.append(torch.from_numpy(gt_project_j2d_img).permute(2,0,1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2,0,1))
        rend_imgs = make_grid(rend_imgs, nrow=4, padding=2)
        return rend_imgs

    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):

        trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

        # Projection in full frame image coordinate
        points = points + trans_full
        points2d_full = perspective_projection(points, rotation=None, translation=None,
                        focal_length=img_focal, camera_center=img_center)

        # Adjust projected points to crop image coordinate
        # (s.t. 1. we can calculate loss in crop image easily
        #       2. we can query its pixel in the crop
        #  )
        b = scale * 200
        points2d = points2d_full - (center - b[:,None]/2)[:,None,:]
        points2d = points2d * (self.crop_size / b)[:,None,None]

        if return_full:
            return points2d_full, points2d
        else:
            return points2d
        
    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        b      = scale * 200
        cx, cy = center[:,0], center[:,1]            # center of crop
        s, tx, ty = pred_cam.unbind(-1)

        img_cx, img_cy = img_center[:,0], img_center[:,1]  # center of original image
        
        bs = b*s
        tx_full = tx + 2*(cx-img_cx)/bs
        ty_full = ty + 2*(cy-img_cy)/bs
        tz_full = 2*img_focal/bs

        trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
        trans_full = trans_full.unsqueeze(1)

        return trans_full
    
    def bbox_est(self, center, scale, img_focal, img_center):
        # Original image center
        img_cx, img_cy = img_center[:,0], img_center[:,1]

        # Implement CLIFF (Li et al.) bbox feature
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8 
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)  

        return bbox_info
