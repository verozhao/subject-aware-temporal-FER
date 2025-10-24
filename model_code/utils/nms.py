import numpy as np
import torch

from utils.box_utils import jaccard


def nms(dets, thresh):
    
    dets = torch.as_tensor(dets).clone()
    if dets.numel() == 0:
        return []
    
    boxes = dets[:, :4]
    scores = dets[:, 4]
    
    _, order = scores.sort(0, descending=True)
    
    chosen = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            chosen.append(i)
            break
        
        i = order[0].item()
        chosen.append(i)
        current_box = boxes[i:i+1, :]
        remaining_boxes = boxes[order[1:], :]
        ious = jaccard(current_box, remaining_boxes)
        ious = ious.squeeze(0)
        idx = torch.where(ious <= thresh)[0]
        order = order[idx + 1]
        
    return chosen
