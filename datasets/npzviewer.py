import argparse
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show .npz content')
    parser.add_argument(
        '--id',
        required=False,
        default='1000092795',
        type=str,
        help='The ID of an example')
    args = parser.parse_args()

    # load numpy-zipped data
    data = numpy.load(f'{args.id}.jpg.npz')

    # lst is a list of keys
    lst = data.files

    for item in lst:
        print(f'Item = {item}')
        print(f'Shape = {data[item].shape}')
        print(data[item])
