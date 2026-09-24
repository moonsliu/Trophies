import numpy as np
import einops
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import default_collate

from trophies.humans.geometry import rotation_6d_to_matrix as rot6d_to_rotmat
from trophies.humans.preprocessing import TrackDataset, split_contiguous_track

from .modules import SMPLTransformerDecoderHead, temporal_attention
from .vit import vit_huge


class HMR_VIMO(nn.Module):
    def __init__(self, cfg=None, device='cpu', **kwargs):

        super(HMR_VIMO, self).__init__()
        self.device = device
        self.cfg = cfg
        self.crop_size = cfg.IMG_RES
        self.seq_len = 16

        # Backbone
        self.backbone = vit_huge()

        # Space-time memory
        if cfg.MODEL.ST_MODULE:
            hdim = cfg.MODEL.ST_HDIM
            nlayer = cfg.MODEL.ST_NLAYER
            self.st_module = temporal_attention(in_dim=1280+3,
                                                out_dim=1280,
                                                hdim=hdim,
                                                nlayer=nlayer,
                                                residual=True)
        else:
            self.st_module = None

        # Motion memory
        if cfg.MODEL.MOTION_MODULE:
            hdim = cfg.MODEL.MOTION_HDIM
            nlayer = cfg.MODEL.MOTION_NLAYER
            self.motion_module = temporal_attention(in_dim=144+3,
                                                    out_dim=144,
                                                    hdim=hdim,
                                                    nlayer=nlayer,
                                                    residual=False)
        else:
            self.motion_module = None

        # SMPL Head
        self.smpl_head = SMPLTransformerDecoderHead()

        self.register_buffer('initialized', torch.tensor(False))


    def forward(self, batch, **kwargs):
        image  = batch['img']
        center = batch['center']
        scale  = batch['scale']
        img_focal = batch['img_focal']
        img_center = batch['img_center']
        # estimate focal length, and bbox
        bbox_info = self.bbox_est(center, scale, img_focal, img_center)

        # backbone
        with torch.amp.autocast(device_type=image.device.type, enabled=image.is_cuda):
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
        pred_pose, pred_shape, pred_cam = self.smpl_head(feature)
        # smpl motion module
        if self.motion_module is not None:
            bb = einops.rearrange(bbox_info, '(b t) c -> b t c', t=16)
            pred_pose = einops.rearrange(pred_pose, '(b t) c -> b t c', t=16)
            pred_pose = torch.cat([pred_pose, bb], dim=2)

            pred_pose = self.motion_module(pred_pose)
            pred_pose = einops.rearrange(pred_pose, 'b t c -> (b t) c')

        out = {
            "pred_cam": pred_cam,
            "pred_pose": pred_pose,
            "pred_shape": pred_shape,
            "pred_rotmat": rot6d_to_rotmat(pred_pose).reshape(-1, 24, 3, 3),
        }
        out["trans_full"] = self.get_trans(
            pred_cam, center, scale, img_focal, img_center
        )
        return out, None


    def inference(self, imgfiles, boxes, img_focal=None, img_center=None, valid=None, frame=None, device='cuda'):
        nfile = len(imgfiles)
        if valid is None:
            valid = np.ones(nfile, dtype=bool)
        if frame is None:
            frame = np.arange(nfile)

        if isinstance(imgfiles, list):
            imgfiles = np.array(imgfiles)

        frame = frame[valid]
        boxes = boxes[valid]
        frame_chunks, boxes_chunks = split_contiguous_track(frame, boxes, min_length=16)

        if len(frame_chunks) == 0:
            return

        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []
        frame = []

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = imgfiles[frame_ck]
            results = self.inference_chunk(
                img_ck,
                boxes_ck,
                img_focal=img_focal,
                img_center=img_center,
                device=device,
            )

            pred_cam.append(results['pred_cam'])
            pred_pose.append(results['pred_pose'])
            pred_shape.append(results['pred_shape'])
            pred_rotmat.append(results['pred_rotmat'])
            pred_trans.append(results['pred_trans'])
            frame.append(torch.from_numpy(frame_ck))

        results = {'pred_cam': torch.cat(pred_cam),
                'pred_pose': torch.cat(pred_pose),
                'pred_shape': torch.cat(pred_shape),
                'pred_rotmat': torch.cat(pred_rotmat),
                'pred_trans': torch.cat(pred_trans),
                'frame': torch.cat(frame)}

        return results


    def inference_chunk(self, imgfiles, boxes, img_focal, img_center, device='cuda'):
        db = TrackDataset(
            imgfiles, boxes, image_focal=img_focal, image_center=img_center, dilation=1.2
        )

        # Results
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []

        items = []
        for i in tqdm(range(len(db))):
            item = db[i]
            items.append(item)

            if len(items) < 16:
                continue
            elif len(items) == 16:
                batch = default_collate(items)
            else:
                items.pop(0)
                batch = default_collate(items)

            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                out, _ = self.forward(batch)

            if len(db) == 16:
                out = {k:v for k,v in out.items()}
            elif i == 15:
                out = {k:v[:9] for k,v in out.items()}
            elif i == len(db) - 1:
                out = {k:v[8:] for k,v in out.items()}
            else:
                out = {k:v[[8]] for k,v in out.items()}

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
