# Raw data

The _flick30k_raw_ and _referit_raw_ folders contain two subfolders, namely
_out_bu_ and _out_ewiser_ with information about images extracted with bottom-up
attention and text extracted with ewiser respectively.

Both folders from _flick30k_raw_ and _referit_raw_ have more or less the same
structure with very litte differences.

## out_bu

This folder contains a numpy-zipped file for every example (i.e. for every
image-sentence pair). Each example is identified by an unique identifier, e.g.
`1000092795` for flickr or `10000` for referit. Obviously, each example is
described by an image and a caption (possibly more than one). (See
[flickr30k.md](flickr30k.md) for more information).

The numpy-zipped file containing information on image is named with the id of
the example followed by `.jpg.npz`, e.g. `1000092795.jpg.npz` or `10000.jpg.npz`
for flickr and referit respectively. This file contains a Python dictionray with
following information:

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

You can see the content of the file with `npzviewer.py`, a simple program
written for this purposes.

```sh
python npzviewer.py --file 1000092795.jpg.npz  # for flicker example
python npzviewer.py --file 10000.jpg.npz       # for referit example
```

The code simply parses the argument, loads the file as a numpy zipped file and
gets data from the dict.

```py
# npzviewer.py

import argparse
import numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show .npz content')
    parser.add_argument('--file', required=False, default='1000092795.jpg.npz', type=str, help='The ID of an example')
    args = parser.parse_args()

    # load numpy-zipped data
    data = numpy.load(args.id)

    # lst is a list of keys
    lst = data.files

    for item in lst:
        print(f'Item = {item}')
        print(f'Shape = {data[item].shape}')
        print(data[item])
```

Sample output ("`...`" means the numpy array, not reported for reading purposes)

```sh
# python npzviewer.py --file 10000.jpg.npz

Item = image_w
Shape = ()
360

Item = cls_prob
Shape = (100, 1601)
...

Item = attr_prob
Shape = (100, 401)
...

Item = bbox
Shape = (100, 4)
...

Item = num_bbox
Shape = ()
100

Item = image_h
Shape = ()
480

Item = x
Shape = (2048, 100)
...
```

## out_ewiser (flickr)

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
      // phrases ...
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
      // noun-phrases ...
    ]
  }

  // sentences ...
]
```

## out_ewiser (referit)

For referit we have the same json file for representing captions and queries,
but in this case, information for a single example is splitted in 5 different
json files (one for caption).

```json
// cat 10000_1.json | python -m json.tool

{
  "img_id": "10000.jpg",
  "ann_id": "10000_1",
  "bbox": [0.0, 78.0, 360.0, 480.0],
  "split": "test",
  "query": ["the ground that's not grass"],
  "ewiser": [
    [
      {
        "chunk": "the ground",
        "head": "ground",
        "token_begin": 4,
        "token_end": 10,
        "synsets": [
          "land.n.02",
          "land.n.04",
          "earth.n.02",
          "soil.n.02",
          "ground.n.09",
          "turf.n.01",
          "geological_formation.n.01",
          "earth.v.02",
          "growth.n.07",
          "property.n.05"
        ],
        "offsets": [
          "wn:09335240n",
          "wn:09334396n",
          "wn:14842992n",
          "wn:14844693n",
          "wn:03462747n",
          "wn:09463919n",
          "wn:09287968n",
          "wn:01292727v",
          "wn:09295338n",
          "wn:04012260n"
        ],
        "scores": [
          0.6774085164070129, 0.2569873034954071, 0.05609188228845596,
          0.007190834265202284, 0.001118004904128611, 0.0004867452662438154,
          0.00038822117494419217, 0.00015768763842061162, 9.171342389890924e-5,
          7.901178469182923e-5
        ],
        "n_synsets": 10
      }
    ]
  ]
}
```

Try this on your own:

```sh
cat 10000_1.txt.json | python -m json.tool
cat 10000_2.txt.json | python -m json.tool
```
