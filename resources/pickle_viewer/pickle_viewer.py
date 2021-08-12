import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A viewer for .pickle files.")
    parser.add_argument("--keys-only", action="store_true", required=False,
                        default=False, help="Whether to visualize only keys")
    parser.add_argument("--file", dest="file", required=False,
                        default="1000092795_0.pickle", help="A .pickle file")

    args = parser.parse_args()

    file = args.file
    keys_only = args.keys_only

    with open(file, "rb") as f:
        x = pickle.load(f)

    for k, v in x.items():
        if keys_only:
            print(k)
        else:
            print(f"{k} = {v}")
