import torch
import torch.nn.functional as F


def tlbr2tlhw(boxes):
    top_left = boxes[..., :2]
    height_width = boxes[..., 2:] - boxes[..., :2]
    return torch.cat([top_left, height_width], dim=-1)


if __name__ == "__main__":
    # assume that we predict 3 bounding box per image and we have a batch of 1
    # bounding box are represented through two points: top left and bottom right corner
    # for each bounding box we have two features

    b_box_1, b_box_2, b_box_3 = [0., 0., 10., 10.], [5., 5., 15., 15.], [30., 30., 60., 60.]

    predicted_bounding_boxes = torch.Tensor([[b_box_1, b_box_2, b_box_3]])              # [b, n_bbox, 4]
    predicted_bounding_boxes_features = torch.Tensor([[[1., 2.], [3., 4.], [5., 6.]]])  # [b, n_bbox, n_feat]

    # get image features

    predicted_bounding_boxes_features_norm = \
        F.normalize(predicted_bounding_boxes_features, p=1, dim=-1)                     # [b, n_bbox, n_feat]

    image_x_area = tlbr2tlhw(predicted_bounding_boxes)                                  # [b, n_bbox, 4]
    image_x_area = (image_x_area[..., 2] * image_x_area[..., 3]).unsqueeze(-1)          # [b, n_bbox, 1]

    image_x = torch.cat([predicted_bounding_boxes_features_norm,
                         predicted_bounding_boxes,
                         image_x_area], dim=-1)                                         # [b, n_bbox, n_feat + 4 + 1]

    print(image_x.size())
