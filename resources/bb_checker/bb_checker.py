import argparse
import os
import pathlib
import pickle


def progress_bar(current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %% (%d)' %
          (arrow, spaces, percent, current), end='\r')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Scan pickle files and detect errors')
    parser.add_argument('--data-dir', dest='data_dir',
                        type=pathlib.Path, required=True, help="Path to pickle files")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data_dir = args.data_dir

    files = os.listdir(data_dir)
    files = list(filter(lambda f: f.endswith(".pickle"), files))
    files = list(filter(lambda f: "_img" in f, files))
    files = list(map(lambda f: os.path.join(data_dir, f), files))

    n_files = len(files)

    n_boxes_bin = [0] * 101

    for i, fname in enumerate(files):
        progress_bar(i, n_files)

        with open(fname, "rb") as f:
            x = pickle.load(f)

            pred_n_boxes = x["pred_n_boxes"]

            n_boxes_bin[pred_n_boxes] += 1

    for i, x in enumerate(n_boxes_bin):
        if x == 0:
            continue

        print(f"Numero di esempi con {i} bounding box = {x}")
