import argparse
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show .npz content')
    parser.add_argument('--file', required=False,
                        default='1000092795.jpg.npz', type=str, help='The numpy-zipped file')
    args = parser.parse_args()

    # load numpy-zipped data
    data = numpy.load(args.file)

    # lst is a list of keys
    lst = data.files

    for item in lst:
        print(f'Item = {item}')
        print(f'Shape = {data[item].shape}')
        print(data[item])
