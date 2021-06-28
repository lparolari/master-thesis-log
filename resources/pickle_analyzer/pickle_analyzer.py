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
    parser.add_argument('--img-only', dest='img_only', action="store_true",
                        default=False, help="Whether to retrieve only images files")
    parser.add_argument('--fail-fast', dest='fail_fast', action="store_true",
                        default=False, help="Wether to fail after first failure")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data_dir = args.data_dir
    img_only = args.img_only
    fail_fast = args.fail_fast

    files = os.listdir(data_dir)
    files = list(filter(lambda f: f.endswith(".pickle"), files))
    files = list(filter(lambda f: "_img" in f if img_only else True, files))
    files = list(map(lambda f: os.path.join(data_dir, f), files))

    n_files = len(files)
    n_corrupted = 0

    for i, fname in enumerate(files):
        progress_bar(i, n_files)

        try:
            with open(fname, "rb") as f:
                x = pickle.load(f)
        except Exception as e:
            print("\n", end="")
            print(fname, f"got an error (i={i})")
            print(e)
            n_corrupted += 1
            if fail_fast:
                raise

        # if not os.path.isfile(file):
        #     print(file, "does not exists")
        #     continue

        # if not "ewiser_chunks" in x.keys():
        #     print(file, "does not contain ewiser_chunks key")
        #     continue

        # if not x["ewiser_chunks"]:
        #     print(file, "has empty ewiser_chunks")
        #     continue

        # if None in x["ewiser_chunks"]:
        #     print(file, "has None in ewiser_chunks")

        # for k, v in x.items():
        #     print(f"{k} = {v}")

    print(f"\nFound {n_corrupted} corrupted files over {n_files} files")
