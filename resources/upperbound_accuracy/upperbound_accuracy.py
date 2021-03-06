import argparse
import logging
import os
import pathlib
import pickle

import numpy as np


def progress_bar(current, total, bar_length=20):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print('Progress: [%s%s] %d %% (%d)' %
          (arrow, spaces, percent, current), end='\r')


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def xyxy2xywh(x):
    """
    Transform coordinates from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height].
    """
    y = np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def get_iou(box1, box2):
    """
    Computes intersection over union (IoU) for two boxes where each box = [x, y, w, h]

    Parameters
    ----------
    box1 : list
        [x, y, w, h] of first box
    box2 : list
        [x, y, w, h] of second box

    Returns
    -------
    float
        intersection over union for box1 and box2

    Source https://github.com/josiahwang/phraseloceval/blob/master/lib/phraseloc/eval/evaluator.py
    """

    (box1_left_x, box1_top_y, box1_w, box1_h) = box1
    box1_right_x = box1_left_x + box1_w - 1
    box1_bottom_y = box1_top_y + box1_h - 1

    (box2_left_x, box2_top_y, box2_w, box2_h) = box2
    box2_right_x = box2_left_x + box2_w - 1
    box2_bottom_y = box2_top_y + box2_h - 1

    # get intersecting boxes
    intersect_left_x = max(box1_left_x, box2_left_x)
    intersect_top_y = max(box1_top_y, box2_top_y)
    intersect_right_x = min(box1_right_x, box2_right_x)
    intersect_bottom_y = min(box1_bottom_y, box2_bottom_y)

    # compute area of intersection
    # the "0" lower bound is to handle cases where box1 and box2 don't overlap
    overlap_x = max(0, intersect_right_x - intersect_left_x + 1)
    overlap_y = max(0, intersect_bottom_y - intersect_top_y + 1)
    intersect = overlap_x * overlap_y

    # get area of union
    union = (box1_w * box1_h) + (box2_w * box2_h) - intersect

    # return iou
    return intersect * 1.0 / union


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Find upperbound accuracy given a number of predicted boxes.")
    parser.add_argument("--data-dir", type=pathlib.Path,
                        required=True, help="Path to pickle files")
    parser.add_argument("--remove-background-boxes", action="store_true", required=False, default=False,
                        help="Skip bounding box annotated with class `__background__`")

    args = parser.parse_args()

    data_dir = args.data_dir
    remove_background_boxes = args.remove_background_boxes

    logging.info(f"Program started, could take a while...")
    logging.info(
        f"data_dir={data_dir}, remove_background_boxes={remove_background_boxes}")

    ks = [(i + 1) * 10 for i in range(10)]

    files = os.listdir(data_dir)
    files = list(filter(lambda f: f.endswith(".pickle"), files))

    # plan:
    # 1. for each image (id) load all examples
    # 2. compute IoU between all pred boxes and phrases 2 crd
    # 3. for each phrases 2 crd box check whether there is one pred box IoU >= 0.5 and count that example
    # 4. compute the accuracy by # match / # all * 100

    index = list(map(lambda x: x.split("_")[0], files))
    index = list(set(index))
    index = sorted(index)

    n_index = len(index)

    total_match = [0] * len(ks)
    total_examples = [0] * len(ks)
    total_backgrounds = [0] * len(ks)

    try:
        for counter, idx in enumerate(index):
            progress_bar(counter, n_index)

            image_file_with_idx = list(filter(lambda x: x.startswith(
                f"{idx}_") and x.endswith("_img.pickle"), files))
            caption_files_with_idx = list(filter(lambda x: x.startswith(
                f"{idx}_") and not x.endswith("_img.pickle"), files))

            assert not len(
                image_file_with_idx) == 0, "Could not find image file"
            assert not len(
                image_file_with_idx) > 1, "Found number more than one image file"

            if len(caption_files_with_idx) == 0:
                print()
                print(
                    f"Skipping {image_file_with_idx[0]}, found {len(caption_files_with_idx)} caption files.")
                continue

            img_data = load_pickle(os.path.join(
                data_dir, image_file_with_idx[0]))
            caption_data_list = [load_pickle(os.path.join(data_dir, filename))[
                "phrases_2_crd"] for filename in caption_files_with_idx]

            # note: on flick3k we may have more than one ground truth bounding
            # box per example linked with given image, so we need to flatten the list
            # in order to create the following numpy array without truble.
            caption_data_list = (
                [item for sublist in caption_data_list for item in sublist])

            boxes = np.array(img_data["pred_boxes"])
            boxes_gt = np.array(caption_data_list)
            pred_cls_prob = np.array(img_data["pred_cls_prob"])
            is_background = np.argmax(pred_cls_prob, axis=-1) == 0

            # ---

            boxes_xywh = xyxy2xywh(boxes)  # [n_boxes, 4]
            # [n_caption * n_query, 4]
            try:
                boxes_gt_xywh = xyxy2xywh(boxes_gt)
            except IndexError:
                print()
                print(
                    f"Oh god, got IndexError for example with index {idx}. "
                    f"img_file={image_file_with_idx}, caption_files={caption_files_with_idx}. "
                    f"Continuing...")
                continue

            n_boxes = boxes_xywh.shape[0]
            n_gt = boxes_gt_xywh.shape[0]

            # compute iou between boxes ang gt
            iou = np.zeros((n_boxes, n_gt))  # [n_boxes, n_gt]

            for i in range(n_boxes):
                box1 = boxes_xywh[i]

                for j in range(n_gt):
                    box2 = boxes_gt_xywh[j]

                    iou[i, j] = get_iou(box1, box2)

            iou = iou.transpose()  # [n_gt, n_boxes]

            # compute number of matches given a fixed number of available bounding boxes
            for h in range(len(ks)):
                k = ks[h]  # get number of available bounding box

                pred_cls_prob_k = pred_cls_prob[:k]
                is_background_k = is_background[:k]
                iou_k = iou[..., :k]

                if remove_background_boxes:
                    mask = np.expand_dims(is_background_k, 0)
                    mask = np.repeat(mask, iou_k.shape[0], axis=0)

                    iou_k_masked = np.ma.array(iou_k, mask=mask)
                    iou_k_masked = np.ma.filled(iou_k_masked, fill_value=0)
                else:
                    iou_k_masked = iou_k

                ok = iou_k_masked > 0.5
                ok = np.any(ok, axis=-1)

                assert ok.ndim == 1

                n_match = ok.sum()
                n_examples = len(ok)
                n_backgrounds = is_background_k.sum()

                total_match[h] += n_match
                total_examples[h] += n_examples
                total_backgrounds[h] += n_backgrounds
    finally:
        accuracy = [total_match[i] / total_examples[i]
                    * 100 for i in range(len(ks))]
        backgrounds = np.array(total_backgrounds) / \
            (np.array(ks) * n_index) * 100

        print()

        logging.info(f"total_match={total_match}")
        logging.info(f"total_examples={total_examples}")
        logging.info(f"total_backgrounds={total_backgrounds}")
        logging.info(f"accuracy={accuracy}")

        print(
            f"Scanned {n_index} images for a total of {total_examples[-1]} queries.")

        for i in range(len(ks)):
            print(
                f"With {ks[i]} bounding box the upperbound accuracy is {accuracy[i]:4f} %" +
                (f", on average we removed {backgrounds[i]:4f} % background boxes." if remove_background_boxes else ""))
