import torch


def bbox_final_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9, clamp=True):
    """
    Get bounding boxes iou among pairs.
    :param box1: bounding boxes 1
    :param box2: bounding boxes 2
    :param x1y1x2y2: if in this format the coordinates.
    :param GIoU: generalize iou. Default false
    :param DIoU: Distance iou. Default false
    :param CIoU: Complete iou. Default false
    :param eps: epsilon value in order to avoid division by 0
    :param clamp:
    :return: bounding boxes iou.
    """

    # sett all coordinates to min values 0.
    if clamp:
        box1 = torch.clamp(box1, min=0, max=1)
        box2 = torch.clamp(box2, min=0, max=1)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / 3.1415927410125732 ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    # drigoni: sometimes was throwing a division by 0 error.
                    denominator = ((1 + eps) - iou + v)
                    denominator_clear = denominator.masked_fill(denominator == 0, eps)
                    alpha = v / denominator_clear

                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


if __name__ == "__main__":
    w, h = 10, 10
    b1 = torch.Tensor([1/w, 1/h, 3/w, 3/h])
    b2 = torch.Tensor([2/w, 2/h, 4/w, 4/h])

    print(bbox_final_iou(b1, b2, x1y1x2y2=True))
    print(1 / 7)

    bb1 = torch.Tensor([[[1/w, 1/h, 3/w, 3/h], [1/w, 1/h, 3/w, 3/h], [1/w, 1/h, 3/w, 3/h]]])
    bb2 = torch.Tensor([[[2/w, 2/h, 4/w, 4/h], [2/w, 2/h, 4/w, 4/h], [2/w, 2/h, 4/w, 4/h]]])

    print(bbox_final_iou(bb1, bb2, x1y1x2y2=True))
