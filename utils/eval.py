import numpy as np
import torch
from .nms import pth_nms
import ipdb

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def model_eval(dataset_val,net,ovthresh=0.5):
    ipdb.set_trace()

    net.eval()

    with torch.no_grad():
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        confidence = np.zeros((0,))

        num_annotations = 0

        for data in dataset_val:
            img, annotations = data['img'], data['annot']
            center_maps, scale_maps = net(img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            #annotations = annotations.cpu().numpy()
            center_maps = center_maps.cpu().numpy()
            scale_maps = scale_maps.cpu().numpy()
            _,__,pred_center_y,pred_center_x = np.where(center_maps>0.01)
            pred_center_y = pred_center_y[:,np.newaxis].astype(np.float)
            pred_center_x = pred_center_x[:,np.newaxis].astype(np.float)
            pred_h = scale_maps[center_maps > 0.01]
            pred_h = np.exp(pred_h)[:,np.newaxis]
            pred_w = pred_h / 0.41
            scores = center_maps[center_maps > 0.01]
            pred_x1, pred_y1,pred_x2,pred_y2 = pred_center_x-pred_w/2,pred_center_y-pred_h/2,pred_center_x+pred_w/2,pred_center_y+pred_h/2
            pred_boxes = np.concatenate((pred_x1,pred_x2,pred_y1,pred_y2),axis=1)
            keep = pth_nms(pred_boxes,0.5)
            pred_boxes = pred_boxes[keep]

            num_annotations += len(annotations)

            detected_annotations = []

            for d in pred_boxes:
                confidence = np.append(confidence, scores)
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
                if max_overlap >= ovthresh and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)

                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)


        indices = np.argsort(-confidence)

        false_positives = false_positives[indices]
        true_positives = true_positives[indices]


        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)


        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        average_precision = _compute_ap(recall, precision)


        return  recall[-1],precision[-1],average_precision

