from typing import List, Callable, Tuple, Any
import logging
import collections
import torch
import torchtext
import os
import pickle
import json
import spacy
import random


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------- CONCEPTS ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------
import random
import typing as t

Token = t.Any
# unfortunately, we don't really know the type of a tokenizer
Tokenizer = t.Callable[[str], Token]
Document = t.List[Token]
Concept = str
Concepts = t.List[str]
Phrases = t.List[str]
Strategy = t.Callable[[Document, int], Concepts]


def extract_concept(phrases: Phrases, tokenizer: Tokenizer, strategy: Strategy) -> Concepts:
    """
    Extract concepts from `phrases` applying `strategy` on document tokenized by the `tokenizer` and returns
    a list of concepts, one per phrase.

    If there is no extracted concepts we return the concept "background".
    """
    documents = [tokenizer(phrase) for phrase in phrases]
    possible_empty_concepts = [strategy(document, 1) for document in documents]
    concepts = ["background" if len(concept) == 0 else concept[0]
                for concept in possible_empty_concepts]
    return concepts


def get_concepts_by(strategy: str):
    """
    Returns a function which returns a list of concepts by giving a document and a number of concepts to retrieve.
    """
    if strategy == "random":
        def f(document: Document, k: int):
            return get_concepts(random_strategy, document, k)
        return f

    if strategy == "first":
        def f(document: Document, k: int):
            return get_concepts(first_strategy, document, k)
        return f

    raise ValueError(f"Strategy {strategy} is not supported.")


def get_concepts(strategy: Strategy, document: Document, k: int) -> Concepts:
    """
    Apply `strategy` on `document` given the number of concepts to extract `k`.
    """
    return strategy(document, k)


def random_strategy(document: Document, k: int) -> Concepts:
    """
    Extract concepts by randomly picking `k` tokens from `document`.

    If `document` is empty we return empty list, i.e., no concepts.
    """
    if len(document) == 0:
        return []
    concepts = [document[random.randrange(0, len(document))] for _ in range(k)]
    concepts = [concept.text for concept in concepts]
    return concepts


def first_strategy(document: Document, k: int) -> Concepts:
    """
    Extract concepts by always picking the first token from `document`.

    If `document` is empty we return empty list, i.e., no concepts.

    If `k > 1`, then the first element is repeated.
    """
    if len(document) == 0:
        return []
    concepts = [document[0] for _ in range(k)]
    concepts = [concept.text for concept in concepts]
    return concepts


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------- PADDER ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------


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
    max_length_examples = 0
    for example in examples:
        max_length_examples = max(max_length_examples, len(example))
    return max_length_examples


def get_max_length_phrases(examples: IndexedExamples):
    """
    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :return: Maximum number of words in phrases among examples
    """
    max_length_phrases = 0
    for example in examples:
        for phrase in example:
            max_length_phrases = max(max_length_phrases, len(phrase))
    return max_length_phrases


def get_indexed_phrases_per_example(examples: TextualExamples, tokenizer: Tokenizer, vocab: Vocab):
    """
    :param examples: A list of examples where each example is a list of phrases (i.e., twice nested lists)
    :param tokenizer: A tokenizer which takes a string and returns a list of string
    :param vocab: A vocabulary which takes a string and returns an int representing the index for given word
    :return: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    """
    def tokenize_phrases(example): return [
        tokenizer(phrase) for phrase in example]

    def index_phrases(example): return [
        index_tokens(phrase) for phrase in example]

    def index_tokens(phrase): return [vocab[token] for token in phrase]

    tokenized_phrases_per_example = [
        tokenize_phrases(example) for example in examples]
    indexed_phrases_per_example = [index_phrases(
        example) for example in tokenized_phrases_per_example]

    return indexed_phrases_per_example if examples != [[]] else [[[]]]


def get_padded_examples(examples: IndexedExamples, padding_dim: Tuple[int, int, int],
                        padding_value: int,
                        dtype: torch.dtype = torch.long):
    """
    :param examples: A list of vocabulary-indexed phrases per example (i.e., triple nested lists)
    :param padding_dim: A tuple of three ints representing the padding dimension of the resulting tensor
    :param padding_value: An integer representing the padding values used in resulting tensor
    :param dtype: Data type
    :return: A tensor with dim=padding_dim with padded values
    """
    padded_tensor = torch.zeros(*padding_dim, dtype=dtype)
    padded_tensor += padding_value

    for ex, ex_data in enumerate(examples):
        for ph, ph_data in enumerate(ex_data):
            for idx, idx_data in enumerate(ph_data):
                if idx_data is not None:
                    padded_tensor[ex, ph, idx] = idx_data
                else:
                    logging.error(
                        f"Indexed word is none at [{ex}, {ph}, {idx}] within phrase {ph_data}")
                    logging.debug(f"{ex_data}")

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
    indexed_phrases_per_example = get_indexed_phrases_per_example(
        examples, tokenizer=tokenizer, vocab=vocab)

    n_examples = get_number_examples(examples)
    max_length_examples = get_max_length_examples(indexed_phrases_per_example)
    max_length_phrases = get_max_length_phrases(indexed_phrases_per_example)

    return get_padded_examples(indexed_phrases_per_example,
                               padding_value=padding_value,
                               padding_dim=(n_examples, max_length_examples, max_length_phrases))
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def load_json(file):
    with open(file, "r") as json_file:
        data = json.load(json_file)
    return data


def load_pickle(file, decompress=True):
    with open(file, "rb") as f:
        if decompress:
            data = pickle.load(f)
        else:
            data = f.read()
    return data


def progress_bar(current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %% (%d)' %
          (arrow, spaces, percent, current), end='\r')


path = "/home/lparolar/Projects/VTKEL-solver"
# path = "/home/2/2019/lparolar/Thesis/VTKEL-solver"
# path = "/mnt/pegasus/home/lparolari/Thesis/data/referit_1"

if __name__ == "__main__":
    import logging
    logging.basicConfig(filename="log3.txt")
    vocab = torchtext.vocab.Vocab(
        collections.Counter(load_json(os.path.join(
            f"{path}/data/referit_raw/vocab.json"))),
        specials=["<pad>"],
        specials_first=True)

    tokenizer = torchtext.data.utils.get_tokenizer(
        "spacy", language="en_core_web_sm")

    spacy = spacy.load("en_core_web_sm")

    files = os.listdir(f"{path}/data/referit_raw/preprocessed")
    files = [os.path.join(path, "data/referit_raw/preprocessed", f)
             for f in files]

    n_files = len(files)

    i_neg = random.randrange(0, n_files)

    for i, f in enumerate(files):
        progress_bar(i, n_files)

        if "_img" in f:
            continue

        # data
        data = load_pickle(f)

        sentence = data['sentence']
        phrases = data['phrases']
        chunks = data['ewiser_chunks']
        chunks_negative = load_pickle(files[i_neg])['ewiser_chunks']

        # indexed data
        indexed_sentences = get_indexed_phrases_per_example(
            [sentence], tokenizer=tokenizer, vocab=vocab)
        indexed_phrases = get_indexed_phrases_per_example(
            phrases, tokenizer=tokenizer, vocab=vocab)
        indexed_chunks = get_indexed_phrases_per_example(
            chunks, tokenizer=tokenizer, vocab=vocab)
        indexed_negative_chunks = get_indexed_phrases_per_example(
            chunks_negative, tokenizer=tokenizer, vocab=vocab)

        indexed_data = [indexed_phrases,
                        indexed_chunks, indexed_negative_chunks]

        # dims
        n_examples = max(map(get_number_examples, indexed_data))
        max_length_examples = max(map(get_max_length_examples, indexed_data))

        # *** CONCEPTS
        concepts = extract_concept(
            phrases=data["ewiser_chunks"], tokenizer=spacy, strategy=get_concepts_by("random"))

        indexed_concepts = get_indexed_phrases_per_example(
            concepts, tokenizer, vocab)

        try:
            concepts, concepts_mask = get_padded_examples(indexed_concepts, padding_dim=(
                n_examples, max_length_examples, 1), padding_value=0)
        except IndexError as e:
            print()
            print("ERROR")
            print("--------------------------------------------")
            print(e)
            print("--------------------------------------------")
            print("EWISER CHUNKS")
            print(data["ewiser_chunks"])
            print("--------------------------------------------")
            print("INDEXED CONCEPTS")
            print(indexed_concepts)
            raise

    ##
    print()
    print("No error found :)")
