import collections
import json
from typing import List, Callable, Tuple, Any

import torch
import torchtext

vocab_path = "resources/vocab.json"

TextualExamples = List[List[str]]
IndexedExamples = List[List[List[int]]]
Tokenizer = Callable[[str], List[str]]
Vocab = torchtext.vocab.Vocab


def get_number_examples(examples: List[Any]):
    """
    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :return: Number of examples
    """
    return len(examples)


def get_max_length_examples(examples: IndexedExamples):
    """
    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :return: Maximum number of phrases among examples
    """
    return len(max(examples, key=len))


def get_max_length_phrases(examples: IndexedExamples):
    """
    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :return: Maximum number of words in phrases among examples
    """
    return len(max([max(example, key=len) for example in examples], key=len))


def get_indexed_phrases_per_example(examples: TextualExamples, tokenizer: Tokenizer, vocab: Vocab):
    """
    :param examples: A list of examples where each example is a list of phrases (i.e., twice nested lists)
    :param tokenizer: A tokenizer which takes a string and returns a list of string
    :param vocab: A vocabulary which takes a string and returns an int representing the index for given word
    :return: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    """
    def tokenize_phrases(example): return [tokenizer(phrase) for phrase in example]

    def index_phrases(example): return [index_tokens(phrase) for phrase in example]

    def index_tokens(phrase): return [vocab[token] for token in phrase]

    tokenized_phrases_per_example = [tokenize_phrases(example) for example in examples]
    indexed_phrases_per_example = [index_phrases(example) for example in tokenized_phrases_per_example]

    return indexed_phrases_per_example


def get_padded_examples(examples: IndexedExamples, padding_dim: Tuple[int, int, int], padding_value: int):
    """
    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :param padding_dim: A tuple of three ints representing the padding dimension of the resulting tensor
    :param padding_value: An integer representing the padding values used in resulting tensor
    :return: A tensor with dim=padding_dim with padded values
    """
    (n_examples, max_length_examples, max_length_phrases) = padding_dim

    padded_tensor = torch.zeros([n_examples, max_length_examples, max_length_phrases], dtype=torch.int64)
    padded_tensor += padding_value

    for ex, ex_data in enumerate(examples):
        for ph, ph_data in enumerate(ex_data):
            for idx, idx_data in enumerate(ph_data):
                padded_tensor[ex, ph, idx] = idx_data

    mask = padded_tensor != padding_value

    return padded_tensor, mask


def get_phrases_tensor(examples: TextualExamples,
                       tokenizer: Tokenizer,
                       vocab: Vocab,
                       padding_value: int = 0) -> (torch.Tensor, torch.Tensor):
    """
    Get phrases representation with pad.
    This function cam be used as a facade for the module.

    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :param tokenizer: A tokenizer which takes a string and returns a list of string
    :param vocab: A vocabulary which takes a string and returns an int representing the index for given word
    :param padding_value: An integer representing the padding values used in resulting tensor
    :return: A tensor with automatically computed dimension and padded values
    """
    indexed_phrases_per_example = get_indexed_phrases_per_example(examples, tokenizer=tokenizer, vocab=vocab)

    n_examples = get_number_examples(examples)
    max_length_examples = get_max_length_examples(indexed_phrases_per_example)
    max_length_phrases = get_max_length_phrases(indexed_phrases_per_example)

    return get_padded_examples(indexed_phrases_per_example,
                               padding_value=padding_value,
                               padding_dim=(n_examples, max_length_examples, max_length_phrases))


if __name__ == "__main__":
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

    phrase1 = "Two young guys with shaggy hair look at their hands while hanging out in the yard ."
    phrase2 = "Two young , White males are outside near many bushes ."
    example1 = [phrase1, phrase2]
    batch1 = [example1]

    value, mask = get_phrases_tensor(examples=batch1, padding_value=0, tokenizer=get_tokenizer(), vocab=get_vocab())
    print(value.size(), mask.size())
    print(value)
    print(mask)


