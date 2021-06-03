from typing import Tuple

import torch
import torch.nn.functional as F

b_box_11, b_box_12, b_box_13 = [0., 0., 10., 10.], [5., 5., 15., 15.], [30., 30., 60., 60.]
b_box_21, b_box_22, b_box_23 = [100., 100., 110., 110.], [105., 105., 115., 115.], [130., 130., 160., 160.]


def iou(_bbox1: torch.Tensor, _bbox2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.Tensor([[[.95, .02, .03], [.45, .55, .0]], [[.20, .60, .20], [.10, .25, .65]]])
    alignments = torch.Tensor([[0, 1], [0, 2]])
    return scores, alignments


def iou_semantic(_iou_scores_proposal, _align_proposal2gt, _predicted_boxes_classifier_probability, _threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    # for laziness we do not hardcode other values, instead we return those hardcoded in `iou` function
    return iou(torch.Tensor(), torch.Tensor())


def complete_iou(_predicted_regression, _phrases_boxes_gt_prime):
    scores, _ = iou(torch.Tensor(), torch.Tensor())
    return scores


def classification_loss(_predicted_logits, _phrases_mask_int, _iousem_scores_proposal):
    target_probabilities = F.normalize(_iousem_scores_proposal, p=1, dim=2)
    predicted_probabilities = F.log_softmax(_predicted_logits, dim=2)  # [b, n_ph_max, n_pred_boxes]

    kl_loss = torch.nn.KLDivLoss(reduction="none")(predicted_probabilities, target_probabilities)  # [b, n_ph_max, n_pred_boxes]
    kl_loss = kl_loss * _phrases_mask_int  # [b, max_n_ph, n_pred_box]
    kl_loss = torch.sum(kl_loss) / _phrases_mask_int.sum()  # []

    return kl_loss


def regression_loss(_predicted_regression, _phrases_mask_int, _iousem_scores_proposal, _phrases_boxes_gt_prime, _eps):
    iousem_scores_prop_norm = _iousem_scores_proposal / (torch.max(_iousem_scores_proposal, dim=-1, keepdim=True)[0] + _eps)
    # calculates iou
    iou_scores = complete_iou(_predicted_regression, _phrases_boxes_gt_prime)
    iou_scores = (1.0 - iou_scores) * iousem_scores_prop_norm
    iou_scores = iou_scores * _phrases_mask_int
    iou_scores = torch.sum(iou_scores) / _phrases_mask_int.sum()
    return iou_scores


if __name__ == "__main__":

    b = 2
    n_ph_max = 2
    n_words_max = 5
    n_pred_boxes = 3
    n_classes = 4

    # the predicted bounding boxes from object detector
    predicted_boxes = torch.Tensor(
        [[b_box_11, b_box_12, b_box_13], [b_box_21, b_box_22, b_box_23]])  # [b, n_pred_boxes, 4]

    # probabilities for classes from object detector for each bounding box
    predicted_boxes_classifier_probability = [[[.75, .15, .5, .5]], [.25, .35, .20, .20],
                                              [.0, .10, .80, .10]]  # [b, n_pred_boxed, n_classes]

    # the ground truth for each phrase with its bounding box coordinates
    phrases_boxes_gt = torch.Tensor(
        [[[15, 15, 30, 30], [40, 40, 50, 50]], [[115, 115, 130, 130], [140, 140, 150, 150]]])  # [b, n_ph_max, 4]
    phrases_mask = torch.Tensor([[[True, True, True, True, True], [True, True, True, False, False]],
                                 [[True, True, True, True, False],
                                  [False, False, False, False, False]]])  # [b, n_ph_max, n_words_max]
    phrases_mask_int = torch.any(phrases_mask, dim=-1, keepdim=True).type(torch.int32)  # [b, n_ph_max, 1]

    # predicted probabilities for each phrase for each bounding box
    predicted_logits = torch.Tensor(
        [[[.98, .01, .01], [.85, .10, .5]], [[.50, .35, .15], [.10, .25, .65]]])  # [b, n_ph_max, n_pred_boxed]

    # regression on bounding box coordinates for each phrase for each bounding box
    predicted_regression = torch.Tensor([[[[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]],
                                          [[40, 40, 40, 40], [50, 50, 50, 50], [60, 60, 60, 60]]],
                                         [[[110, 110, 110, 110], [120, 120, 120, 120], [130, 130, 130, 130]],
                                          [[140, 140, 140, 140], [150, 150, 150, 150],
                                           [160, 160, 160, 160]]]])  # [b, n_ph_max, n_pred_boxed, 4]

    # ************************************

    # for some dimension for this tensors
    phrases_boxes_gt_prime = phrases_boxes_gt.unsqueeze(2).repeat(1, 1, n_pred_boxes, 1)  # [b, n_ph_max, n_pred_boxes, 4]
    predicted_boxes_prime = predicted_boxes.unsqueeze(1).repeat(1, n_ph_max, 1, 1)  # [b, n_ph_max, n_pred_boxes, 4]

    # Compute IoU between predicted proposals and the gt bounding box (i.e in the paper U_jz)
    #   iou_scores_proposal [b, n_ph_max, n_pred_boxes], a tensor with iou scores between phrase against bounding box
    #   align_proposal2gt [b, n_ph_max], the arg max over n_pred_boxes, i.e., the best bounding box for a given phrase
    iou_scores_proposal, align_proposal2gt = iou(phrases_boxes_gt_prime, predicted_boxes_prime)

    # Compute Semantic IoU (i.e in the paper U_jz^*)
    iousem_scores_proposal, align_iousem2gt = iou_semantic(iou_scores_proposal, align_proposal2gt, predicted_boxes_classifier_probability, _threshold=0.5)

    # Compute the classification loss and regression loss (L_g and L_c in the paper)
    grounding_loss = classification_loss(predicted_logits, phrases_mask_int, iousem_scores_proposal)
    coordinates_loss = regression_loss(predicted_regression, phrases_mask_int, iousem_scores_proposal, phrases_boxes_gt_prime, _eps=0.5)

    print(grounding_loss, coordinates_loss)
