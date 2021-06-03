import torch


def fuse(phrases_x, image_x):
    predicted_text_representation = phrases_x.unsqueeze(2)                                                              # [b, n_ph_max, 1, n_text_feat]
    predicted_text_representation = predicted_text_representation.repeat(1, 1, num_predicted_boxes, 1)                  # [b, n_ph_max, n_pred_boxes, n_text_feat]

    predicted_image_representation = image_x.unsqueeze(1)                                                               # [b, 1, n_pred_boxes, n_img_feat]
    predicted_image_representation = predicted_image_representation.repeat(1, num_phrases_max, 1, 1)                    # [b, n_ph_max, n_pred_boxes, n_img_feat]

    predicted_representation = torch.cat([predicted_text_representation, predicted_image_representation], dim=-1)       # [b, n_ph_max, n_pred_boxed, n_text_feat+n_img_feat] = [2,2,5,3+2]

    return predicted_representation


if __name__ == "__main__":
    num_predicted_boxes = 5
    num_phrases_max = 2

    phrases_x = torch.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # [b, n_ph_max, n_text_feat]
    image_x = torch.Tensor([
        [[i + 101, i + 102] for i in range(num_predicted_boxes)],
        [[i + 201, i + 202] for i in range(num_predicted_boxes)]])                 # [b, n_pred_boxes, n_img_feat]

    predicted_representation = fuse(phrases_x, image_x)                            # [b, n_ph_max, n_pred_boxed, n_text_feat+n_img_feat]

    print(f"predicted_representation.size() = {predicted_representation.size()}")
    print(predicted_representation)
