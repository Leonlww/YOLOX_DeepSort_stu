# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None, cls_id = None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []
    b_pick = []
    h_pick = []
    head_boxes = []
    body_boxes = []
    head_scores = []
    body_scores = []
    head_index_dic = {}
    body_index_dic = {}
    for i in range(len(cls_id)):
        if cls_id[i] == 0:
            body_scores.append(scores[i])
            body_boxes.append(boxes[i])
        if cls_id[i] == 2:
            head_scores.append(scores[i])
            head_boxes.append(boxes[i])

    head_scores = np.array(head_scores)
    body_scores = np.array(body_scores)
    head_boxes = np.array(head_boxes)
    body_boxes = np.array(body_boxes)
    if len(head_boxes)>0:
        head_boxes = np.stack(head_boxes,axis=0)
    if len(body_boxes)>0:
        body_boxes = np.stack(body_boxes,axis=0)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = boxes[:, 2] + boxes[:, 0]
    by2 = boxes[:, 3] + boxes[:, 1]
    b_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)

    hx1 = boxes[:, 0]
    hy1 = boxes[:, 1]
    hx2 = boxes[:, 2] + boxes[:, 0]
    hy2 = boxes[:, 3] + boxes[:, 1]
    h_area = (hx2 - hx1 + 1) * (hy2 - hy1 + 1)

    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)
    if body_scores is not None:
        b_idxs = np.argsort(body_scores)
    else:
        b_idxs = np.argsort(by2)

    if head_scores is not None:
        h_idxs = np.argsort(head_scores)
    else:
        h_idxs = np.argsort(hy2)

    for index in idxs:
        for bi in b_idxs:
            if scores[index] == body_scores[bi]:
                body_index_dic[bi] = index
        for hi in h_idxs:
            if scores[index] == head_scores[hi]:
                head_index_dic[hi] = index
    while len(b_idxs) > 0:
        last = len(b_idxs) - 1
        i = b_idxs[last]
        b_pick.append(i)

        bxx1 = np.maximum(bx1[i], bx1[b_idxs[:last]])
        byy1 = np.maximum(by1[i], by1[b_idxs[:last]])
        bxx2 = np.minimum(bx2[i], bx2[b_idxs[:last]])
        byy2 = np.minimum(by2[i], by2[b_idxs[:last]])

        w = np.maximum(0, bxx2 - bxx1 + 1)
        h = np.maximum(0, byy2 - byy1 + 1)

        overlap = (w * h) / b_area[b_idxs[:last]] # IOU 

        b_idxs = np.delete(
            b_idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))


    while len(h_idxs) > 0:
        last = len(h_idxs) - 1
        i = h_idxs[last]
        h_pick.append(i)

        hxx1 = np.maximum(bx1[i], bx1[h_idxs[:last]])
        hyy1 = np.maximum(by1[i], by1[h_idxs[:last]])
        hxx2 = np.minimum(bx2[i], bx2[h_idxs[:last]])
        hyy2 = np.minimum(by2[i], by2[h_idxs[:last]])

        w = np.maximum(0, hxx2 - hxx1 + 1)
        h = np.maximum(0, hyy2 - hyy1 + 1)

        overlap = (w * h) / h_area[h_idxs[:last]] # IOU 

        h_idxs = np.delete(
            h_idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))


    for body_i in b_pick:
        pick.append(body_index_dic[body_i])
    for head_i in h_pick:
        pick.append(head_index_dic[head_i])
    

    return pick
