import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A viewer for .pickle files.")
    parser.add_argument("--file", dest="file", required=False,
                        default="1000092795_0.pickle", help="A .pickle file")
    parser.add_argument("--show-content", action="store_true",
                        default=False, help="Show full content")

    args = parser.parse_args()

    file = args.file
    show_content = args.show_content

    with open(file, "rb") as f:
        x = pickle.load(f)

    print(x.keys())

    if show_content is True:
        print(x)
