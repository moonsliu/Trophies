from pathlib import Path

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torchvision

from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as masktool

from trophies.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig


if torch.cuda.is_available():
    autocast = torch.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def video2frames(vidfile, save_folder):
    """ Convert input video to images """
    count = 0
    cap = cv2.VideoCapture(vidfile)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite(f'{save_folder}/{count:04d}.jpg', frame)
            count += 1
        else:
            break
    cap.release()
    return count


def detect_segment_track(imgfiles, out_path, thresh=0.5, min_size=None,
                         device='cuda', save_vos=True):
    """Detect, segment, and track people for masked camera estimation.

    Detection: ViTDet.
    Segmentation: SAM.
    Tracking: DEVA-Track-Anything.
    """
    from .deva_track import get_deva_tracker, track_with_mask, flush_buffer

    cfg_path = PROJECT_ROOT / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    sam = sam_model_registry["vit_h"](checkpoint=str(PROJECT_ROOT / "checkpoints" / "sam_vit_h_4b8939.pth"))
    _ = sam.to(device)
    predictor = SamPredictor(sam)

    vid_length = len(imgfiles)
    deva, result_saver = get_deva_tracker(vid_length, out_path)

    masks_ = []
    boxes_ = []
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        with torch.no_grad():
            with autocast('cuda'):
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > thresh)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                confs = det_instances.scores[valid_idx].cpu().numpy()

                boxes = np.hstack([boxes, confs[:, None]])
                boxes = arrange_boxes(boxes, mode='size', min_size=min_size)

        if len(boxes)>0:
            with autocast('cuda'):
                predictor.set_image(img_cv2, image_format='BGR')

                bb = torch.tensor(boxes[:, :4]).cuda()
                bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bb,
                    multimask_output=False
                )
                scores = scores.cpu()
                masks = masks.cpu().squeeze(1)
                mask = masks.sum(dim=0)
        else:
            mask = np.zeros(img_cv2.shape[:2], dtype=np.uint8)

        if len(boxes)>0 and (boxes[:, -1] > 0.80).sum()>0:
            # DEVA receives only high-confidence detections.
            track_valid = boxes[:, -1] > 0.80
            masks_track = masks[track_valid]
            scores_track = scores[track_valid]
        else:
            masks_track = torch.zeros([1,img_cv2.shape[0],img_cv2.shape[1]])
            scores_track = torch.zeros([1])

        with autocast('cuda'):
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            track_with_mask(deva, masks_track, scores_track, img_rgb,
                            imgpath, result_saver, t, save_vos)

        mask_bit = masktool.encode(np.asfortranarray(mask > 0))
        masks_.append(mask_bit)
        boxes_.append(boxes)

    with autocast('cuda'):
        flush_buffer(deva, result_saver)
    result_saver.end()

    # Convert DEVA annotations into per-identity temporal tracks.
    vjson = result_saver.video_json
    ann = vjson['annotations']
    iou_thresh = 0.5
    conf_thresh = 0.5

    tracks = {}
    for frame in ann:
        seg = frame['segmentations']
        file = frame['file_name']
        frame = int(file.split('.')[0])
        for subj in seg:
            idx = subj['id']
            msk = subj['rle']
            msk = torch.from_numpy(masktool.decode(msk))[None]

            # match tracked segment to detections
            det_boxes = boxes_[frame]
            if len(det_boxes)>0:
                seg_box = torchvision.ops.masks_to_boxes(msk)
                iou = box_iou(det_boxes, seg_box)
                max_iou, max_id = iou.max(), iou.argmax()
                max_conf = det_boxes[max_id, -1]
            else:
                max_iou = max_conf = 0

            if max_iou>iou_thresh and max_conf>conf_thresh:
                det = True
                det_box = det_boxes[[max_id]]
            else:
                det = False
                det_box = np.zeros([1, 5])

            subj['frame'] = frame
            subj['det'] = det
            subj['det_box'] = det_box
            subj['seg_box'] = seg_box.numpy()

            if idx in tracks:
                tracks[idx].append(subj)
            else:
                tracks[idx] = [subj]

    tracks = np.array(tracks, dtype=object)
    masks_ = np.array(masks_, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, masks_, tracks



def arrange_boxes(boxes, mode='size', min_size=None):
    """ Helper to re-order boxes """
    # Left2right priority
    if mode == 'left2right':
        cx = (boxes[:,2] - boxes[:,0]) / 2 + boxes[:,0]
        boxes = boxes[np.argsort(cx)]
    # size priority
    elif mode == 'size':
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        area = w*h
        boxes = boxes[np.argsort(area)[::-1]]
    # confidence priority
    elif mode == 'conf':
        conf = boxes[:,4]
        boxes = boxes[np.argsort(conf)[::-1]]
    # filter boxes by size
    if min_size is not None:
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        valid = np.stack([w, h]).max(axis=0) > min_size
        boxes = boxes[valid]

    return boxes


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if type(box1) == np.ndarray:
        box1 = torch.from_numpy(box1)
    if type(box2) == np.ndarray:
        box2 = torch.from_numpy(box2)
    box1 = box1[:, :4]
    box2 = box2[:, :4]

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
