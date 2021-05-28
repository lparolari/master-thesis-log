"""
This script is just to investigate what the `create_embeddings_network` function does.
Basically, it builds the embeddings for the words in vocabulary extracted from our domain (i.e., vocab.json).
The embeddings are taken from GloVe if word in vocabulary exists in GloVe, otherwise the embedding is learned
through a neural network.

The function [create_embeddings_network](
https://github.com/lparolari/Loss_VT_Grounding/blob/0f5627dc9dda6edbe2be130b1eaf5a3fc98cc171/model_code/model.py#L165
) has the following prototype:

```
create_embeddings_network(embedding, vocab, text_emb_size, freeze=False)
```

where

* `embedding` means the type of the embedding, i.e. "glove" or "nn"
* `vocab` is an instance of torchtext.vocab.Vocab, which is the vocabulary created by the json dict
* `text_emb_size` defines the embedding size and is usually initialized at 300 in the trainer
* `freeze` sets whether to learn the representation with gradients or not
"""

import argparse
import collections
import json
import torchtext


def load_json(file):
    with open(file, "r") as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-debug", action="store_false", dest="is_debug", required=False)
    parser.add_argument("--vocab", dest="vocab_path", required=False, default="resources/vocab.json")
    args = parser.parse_args()

    is_debug = args.is_debug
    vocab_path = args.vocab_path

    # We first load the vocabulary extracted from captions. Vocab is saved as a JSON object so we first
    # parse that file and we save the output.
    # Please note that the parsed JSON could be anything (dict, list, string, number...), in our case is a dict.
    vocab_dict = load_json(vocab_path)

    if is_debug:
        print(f"1. The dictionary contains {len(vocab_dict.keys())} words")

    # Then we store this dict in a specific collection, which is a sub-class of dict where keys are objects in our
    # case strings and values are integers. Note: we need the counter for creating a torch vocabulary.
    vocab_counter = collections.Counter(vocab_dict)

    if is_debug:
        print(
            f"2. Vocab counter is a dict, here we list the first three keys: {list(vocab_counter.keys())[:3]}")

    # We create a vocabulary from torch. The Vocab class assign an index for each word such that we can represent
    # numerically the word. For example, with vocab.itos[i] we can retrieve the word with index i, instead with
    # vocab.stoi[s] we can retrieve the index of the word s.
    # Please note that we do not have .vectors on this vocab as it isn't trained.
    vocab = torchtext.vocab.Vocab(vocab_counter, specials=[
                                  '<pad>'], specials_first=True)

    if is_debug:
        print(f"3. The word with index 21362 is {vocab.itos[21362]}, while the word \"zero\" has index "
              f"{vocab.stoi['zero']}.")

    # Let's get GloVe embeddings. Those embedding are built from a large vocabulary (we have more than 2 million
    # words) and they take into account the co-occurrence statistics. For example, man is near sir and both this two
    # are far from women, but farther wrt animals. Please note that we can retrieve the embedding vector for each
    # word with .vectors[idx]
    # https://nlp.stanford.edu/projects/glove/
    glove_embeddings = torchtext.vocab.GloVe('840B', dim=300)

    if is_debug:
        print(f"4. GloVe, three examples {list(glove_embeddings.stoi.keys())[10:13]}. An example of sliced vector for"
              f" the word 'is' with index {glove_embeddings.stoi['is']}: "
              f"{glove_embeddings.vectors[glove_embeddings.stoi['is']][:3]}")
