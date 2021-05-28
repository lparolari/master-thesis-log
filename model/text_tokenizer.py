import collections
import json
import torch
import torchtext


vocab_path = "resources/vocab.json"
padding_value = 0


def get_phrases_tensor(batch):
    # batch is a list of examples, where a list of example is a list of phrases
    tokenizer = get_tokenizer()
    vocab = get_vocab()

    # batch -> examples -> phrases -> tokens

    def tokenize_phrases(example): return [tokenizer(phrase) for phrase in example]
    def index_phrases(example): return [index_tokens(phrase) for phrase in example]
    def index_tokens(phrase): return [vocab[token] for token in phrase]

    tokenized_phrases_per_example = [tokenize_phrases(example) for example in batch]
    indexed_phrases_per_example = [index_phrases(example) for example in tokenized_phrases_per_example]

    # Padding normalizes number of phrases per example and also phrases length.
    # In this example we defined two phrases from the same example (i.e. max_phrases_for_example=2) and
    # a single example (i.e. batch_size=1).
    batch_size = len(indexed_phrases_per_example)
    max_phrases_for_example = len(max(indexed_phrases_per_example, key=len))
    max_length_phrases = len(max([max(example, key=len) for example in indexed_phrases_per_example], key=len))

    # print(f"batch_size={batch_size}, "
    #       f"max_phrases_for_example={max_phrases_for_example}, "
    #       f"max_length_phrases={max_length_phrases}")

    padded_tensor = torch.zeros([batch_size, max_phrases_for_example, max_length_phrases], dtype=torch.int64)
    padded_tensor += padding_value

    for es, es_data in enumerate(indexed_phrases_per_example):
        for ph, ph_data in enumerate(es_data):
            for idx, idx_data in enumerate(ph_data):
                padded_tensor[es, ph, idx] = idx_data

    mask = padded_tensor != padding_value

    return padded_tensor, mask


def get_vocab():
    vocab_dict = load_json(vocab_path)
    vocab_counter = collections.Counter(vocab_dict)
    return torchtext.vocab.Vocab(vocab_counter, specials=['<pad>'], specials_first=True)


def get_tokenizer():
    return torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')


def load_json(file):
    with open(file, "r") as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    phrase1 = "Two young guys with shaggy hair look at their hands while hanging out in the yard ."
    phrase2 = "Two young , White males are outside near many bushes ."
    example1 = [phrase1, phrase2]
    batch1 = [example1]

    print(get_phrases_tensor(batch1))
