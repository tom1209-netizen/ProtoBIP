"""
CAM generation and visualization utilities
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from .crf import DenseCRF
from .evaluate import ConfusionMatrixAllClass
from .hierarchical_utils import merge_to_parent_predictions, merge_subclass_cams_to_parent
from .pyutils import AverageMeter


def get_seg_label(cams, inputs, label):
    """Generate segmentation labels from CAMs"""
    with torch.no_grad():
        b, c, h, w = inputs.shape
        label = label.view(b, -1, 1, 1).cpu().data.numpy()
        cams = cams.cpu().data.numpy()
        cams = np.maximum(cams, 0)
        
        # Normalize CAMs to [0,1]
        channel_max = np.max(cams, axis=(2, 3), keepdims=True)
        channel_min = np.min(cams, axis=(2, 3), keepdims=True)
        cams = (cams - channel_min) / (channel_max - channel_min + 1e-6)
        cams = cams * label
        cams = torch.from_numpy(cams).float()
        
        # Resize to input dimensions
        cams = F.interpolate(cams, size=(h, w), mode="bilinear", align_corners=True)
        cam_max = torch.max(cams, dim=1, keepdim=True)[0]
        bg_cam = (1 - cam_max) ** 10
        cam_all = torch.cat([cams, bg_cam], dim=1)

    return cams



def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    """Validation function with test-time augmentation AND CRF post-processing."""
    model.eval()
    avg_meter = AverageMeter()
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)

    crf_processor = DenseCRF()
    MEAN = [0.66791496, 0.47791372, 0.70623304] # Mean values for normalization, specific to the dataset
    STD = [0.1736589,  0.22564577, 0.19820057] # Standard deviation values.
    std_tensor = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean_tensor = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)
    
    tta_transform = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            (cls1, _, _, _, _, _, cls4, _, _, k_list, _) = model(inputs)
            cls1 = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
            cls4 = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)
            cls_loss = cls_loss_func(cls1, cls_label)
            cls4_acc_check = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4_acc_check == cls_label).all(dim=1).float().sum() / cls4_acc_check.shape[0] * 100
            avg_cls_acc4 = ((cls4_acc_check == cls_label).sum(dim=0) / cls4_acc_check.shape[0]).mean() * 100
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss})
            
            cams2 = []
            cams3 = []
            cams4 = []
            for tta_trans in tta_transform:
                augmented_tensor = tta_trans.augment_image(inputs)
                (_, _, _, cam2, _, cam3, _, cam4, _, k_list, _) = model(augmented_tensor)

                cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
                cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
                cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

                cam2 = get_seg_label(cam2, augmented_tensor, cls_label).cuda()
                cam2 = tta_trans.deaugment_mask(cam2).unsqueeze(dim=0)
                cams2.append(cam2)
                
                cam3 = get_seg_label(cam3, augmented_tensor, cls_label).cuda()
                cam3 = tta_trans.deaugment_mask(cam3).unsqueeze(dim=0)
                cams3.append(cam3)

                cam4 = get_seg_label(cam4, augmented_tensor, cls_label).cuda()
                cam4 = tta_trans.deaugment_mask(cam4).unsqueeze(dim=0)
                cams4.append(cam4)

            cams2 = torch.cat(cams2, dim=0).mean(dim=0)
            cams3 = torch.cat(cams3, dim=0).mean(dim=0)
            cams4 = torch.cat(cams4, dim=0).mean(dim=0)

            fuse234 = 0.3 * cams2 + 0.3 * cams3 + 0.4 * cams4

            cam_max = torch.max(fuse234, dim=1, keepdim=True)[0]
            bg_cam = (1 - cam_max) ** 10
            
            full_probs_tensor = torch.cat([fuse234, bg_cam], dim=1)
            
            full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
            probs_np = full_probs_tensor.cpu().numpy()

            img_denorm_tensor = (inputs * std_tensor) + mean_tensor
            img_denorm_tensor = torch.clamp(img_denorm_tensor * 255, 0, 255).byte()
            img_np = img_denorm_tensor.permute(0, 2, 3, 1).cpu().numpy()

            refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
            
            fuse_label234 = torch.from_numpy(refined_mask).long().cuda()
            
            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")
    fuse234_score = fuse234_matrix.compute()[2]
    model.train()
    return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss


def generate_cam(model=None, data_loader=None, cfg=None, cls_loss_func=None):
    model.eval()

    ## Instantiate the CRF processor
    crf_processor = DenseCRF()

    ## Get MEAN/STD for de-normalization
    MEAN = [0.66791496, 0.47791372, 0.70623304] # Mean values for normalization, specific to the dataset
    STD = [0.1736589,  0.22564577, 0.19820057] # Standard deviation values.
    std = torch.tensor(STD, device='cuda').view(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device='cuda').view(1, 3, 1, 1)

    with torch.no_grad():
        for data in tqdm(data_loader,
                         total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data

            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, k_list, _) = model(inputs)

            cam1 = merge_subclass_cams_to_parent(cam1, k_list, method=cfg.train.merge_test)
            cam2 = merge_subclass_cams_to_parent(cam2, k_list, method=cfg.train.merge_test)
            cam3 = merge_subclass_cams_to_parent(cam3, k_list, method=cfg.train.merge_test)
            cam4 = merge_subclass_cams_to_parent(cam4, k_list, method=cfg.train.merge_test)

            cam1 = get_seg_label(cam1, inputs, cls_label).cuda()
            cam2 = get_seg_label(cam2, inputs, cls_label).cuda()
            cam3 = get_seg_label(cam3, inputs, cls_label).cuda()
            cam4 = get_seg_label(cam4, inputs, cls_label).cuda()

            fuse234 = 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
            
            # This is the original, blobby prediction. We will replace this.
            # output_fuse234 = torch.argmax(fuse234, dim=1).long()

            ## Prepare inputs for the CRF
            # The CRF needs a probability map, so we apply softmax to the fused CAMs.
            # We add a background channel as the CRF expects probabilities for all classes including background.
            cam_max = torch.max(fuse234, dim=1, keepdim=True)[0]
            bg_cam = (1 - cam_max) ** 10 # Using your background estimation method
            full_probs_tensor = torch.cat([fuse234, bg_cam], dim=1) # Note: The paper uses 5 classes (4 + background)
            
            num_actual_classes = cfg.dataset.cls_num_classes + 1
            full_probs_tensor = F.softmax(full_probs_tensor, dim=1)
            probs_np = full_probs_tensor.cpu().numpy() # Shape: [B, 5, H, W]

            # De-normalize the input image for the CRF's bilateral filter
            img_denorm_tensor = (inputs * std) + mean
            img_denorm_tensor = torch.clamp(img_denorm_tensor * 255, 0, 255).byte()
            img_np = img_denorm_tensor.permute(0, 2, 3, 1).cpu().numpy() # Shape: [B, H, W, C]

            ## CRF Step 4: Apply the CRF
            refined_mask, _ = crf_processor.process(probs=probs_np, images=img_np)
            # The output `refined_mask` is a numpy array of shape [B, H, W] with integer class labels.

            # Convert the refined mask back to a tensor for saving logic
            output_refined = torch.from_numpy(refined_mask).long()

            PALETTE = [
                [255, 0, 0] ,   # TUM
                [0, 255, 0],   # STR
                [0, 0, 255],   # LYM
                [153, 0, 255], # NEC
                [0, 0, 0],     # BACK - Assuming background is class 4 now
            ]

            for i in range(len(output_refined)):
                pred_mask = Image.fromarray(output_refined[i].cpu().squeeze().numpy().astype(np.uint8)).convert('P')
                flat_palette = [val for sublist in PALETTE for val in sublist]
                pred_mask.putpalette(flat_palette)
                pred_mask.save(os.path.join(cfg.work_dir.pred_dir, name[i])) 
    model.train()
    return
