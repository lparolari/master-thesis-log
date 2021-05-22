# flickr30k_raw

The _flick30k_raw_ folder contains two subfolders, namely _out_bu_ and
_out_ewiser_ with information about images and text respectively.

## out_bu

This folder contains a numpy-zipped file for every example (i.e. for every
image-sentence pair). Each example is identified by an unique identifier, e.g.
`1000092795`. Obviously, each example is composed by an image and a sentence
describing the image. (See [flickr30k.md](flickr30k.md) for more information).

The numpy-zipped file containing information on image is named with the id of
the example followed by `.jpg.npz`, e.g. `1000092795.jpg.npz`. The former
contains a Python dictionray with following information:

- `image_w`, scalar, represents image width;
- `image_h`, scalar, represents image height;
- `cls_prob`, `(100, 1601)` matrix , the probability for each bounding box to be
  one of the 1601 classes;
- `attr_prob`, `(100, 401)` matrix, the probability for each bounding box to
  have one of the 401 attributes;
- `bbox`, `(100, 4)` matrix, position for each bounding box expressed with two
  coordinates: top left corner (x, y) and bottom right corner (x, y);
- `num_bbox`, scalar, number of extracted bounding box;
- `x`, `(2048, 100)` matrix, features for each bounding box;

```py
# npzviewer.py

import argparse
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show .npz content')
    parser.add_argument('--id', required=False, default='1000092795', type=str, help='The ID of an example')
    args = parser.parse_args()

    # load numpy-zipped data
    data = numpy.load(f'{args.id}.jpg.npz')

    # lst is a list of keys
    lst = data.files

    for item in lst:
        print(f'Item = {item}')
        print(f'Shape = {data[item].shape}')
        print(data[item])
```

## out_ewiser

This folder contains a json file for each example, identified by its unique
identifier (e.g. `1000092795`). The json file contains information about the
sentences that describe image. More precisely we have a json array where each
entry represent a single sentence (caption). The sentence is described by a json
object which report the full sentence and information on the noun-phrases in
that sentence with many details from ewiser (such as phrase type, synsets - or
synonims -, a score for each synset, ...).

The following is an example of what this type of files contain. In order to get
this output you can execute

```sh
cat 1000092795.txt.json | python -m json.tool
```

assuming you have python installed.

```json
// 1000092795.txt.json

[
  {
    "sentence": "Two young guys with shaggy hair look at their hands while hanging out in the yard .",
    "phrases": [
      {
        "first_word_index": 0,
        "phrase": "Two young guys",
        "phrase_id": "1",
        "phrase_type": ["people"]
      }
      // ...
    ],
    "ewiser": [
      {
        "chunk": "Two young guys",
        "head": "guys",
        "token_begin": 10,
        "token_end": 14,
        "synsets": [
          "guy.n.01",
          "man.n.01",
          "chap.n.01",
          "boy.n.02",
          "fellow.n.06",
          "male.n.02",
          "boyfriend.n.01",
          "young_buck.n.01",
          "boy.n.04",
          "geezer.n.01"
        ],
        "offsets": [
          "wn:10153414n",
          "wn:10287213n",
          "wn:09908025n",
          "wn:09870926n",
          "wn:10083358n",
          "wn:09624168n",
          "wn:09871364n",
          "wn:10804287n",
          "wn:09637837n",
          "wn:10123711n"
        ],
        "scores": [
          0.9945911765098572, 0.003881721757352352, 0.0009650802821852267,
          0.00021727304556407034, 0.00018178755999542773, 5.252950722933747e-5,
          4.8552210500929505e-5, 4.523143434198573e-5, 8.64600770000834e-6,
          7.889746484579518e-6
        ],
        "n_synsets": 10
      }
      // ...
    ]
  }

  // ...
]
```
