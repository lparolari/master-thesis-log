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
    def tokenize_phrases(example):
        # print("tokenize_phrases", [tokenizer(phrase) for phrase in example])
        return [tokenizer(phrase) for phrase in example]

    def index_phrases(example): return [
        index_tokens(phrase) for phrase in example]

    def index_tokens(phrase): return [vocab[token] for token in phrase]

    # print("tokenized_phrases_per_example", [
    #       tokenize_phrases(example) for example in examples])
    tokenized_phrases_per_example = [
        tokenize_phrases(example) for example in examples]
    # print("indexed_phrases_per_example", [index_phrases(
    #     example) for example in tokenized_phrases_per_example])
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
                    raise ValueError

    mask = padded_tensor != padding_value

    return padded_tensor, mask


def get_phrases_tensor(examples: TextualExamples,
                       tokenizer: Tokenizer,
                       vocab: Vocab,
                       padding_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
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


# path = "/mnt/pegasus/home/lparolari/Thesis/master-thesis-log/resources/index_word_none"
debug = False

path = "/home/lparolar/Projects/VTKEL-solver"
# path = "/home/2/2019/lparolar/Thesis/VTKEL-solver"
# path = "/mnt/pegasus/home/lparolari/Thesis/data/referit_1"

vocab_yago_dict = load_json(os.path.join(
    path, "data/referit_raw/vocab_yago.json"))
vocab_yago = torchtext.vocab.Vocab(collections.Counter(
    vocab_yago_dict), specials=['<pad>'], specials_first=True)


def tokenization_ewiser_entities(data, n_entities=10):
    """
    Use the vocabulary to retrieve entity indexes for the EWISER predictions.
    Moreover, this function orders and filters the entities.
    :param data: list of list entities.
    :return: list of list of indexes.
    """
    keys = ['missing', 'type a', 'type r', 'type s', 'type v']
    results = []
    for raw_ent in data:
        if raw_ent is not None:
            # clean the predictions
            tmp = ['<pad>'] * n_entities
            counter = 0
            for i in range(n_entities):
                if raw_ent[i] is not keys:
                    tmp[counter] = raw_ent[i]
                    counter += 1
            # retrieve index
            tmp = [vocab_yago[i] for i in tmp]
        else:
            tmp = [vocab_yago['missing']] * n_entities
        results.append(tmp)
    return results


"""
Analizza e colleziona tutti gli esempi tali per cui la funzione di padding
trova una parola indicizzata (`idx_data`) = None.

Restituisce un sommario suddiviso per sentence, phrases, chunks, yago_chunks 
e phrases_2_crd dove vengono contati il num di esempi affetti da questo problema
e ne lista i nomi di file.

NOTA: a quanto sappiamo, questo problema succede solo con i chunk, ma gli altri
sono controllti per scrupolo.
"""

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

    errors = [0] * 5
    errors_f = [[]] * 5

    # i_neg = random.randrange(0, n_files)

    for i, f in enumerate(files):
        progress_bar(i, n_files)

        if "_img" in f:
            continue

        # data
        data = load_pickle(f)

        sentence = [data['sentence']]
        phrases = [data['phrases']]
        chunks = [data['ewiser_chunks']]
        phrases_2_crd = [data['phrases_2_crd']]
        yago_entities__ = [data['ewiser_yago_entities']]
        # chunks_negative = load_pickle(files[i_neg])['ewiser_chunks']

        # indexed data
        indexed_sentences = get_indexed_phrases_per_example(
            [sentence], tokenizer=tokenizer, vocab=vocab)
        indexed_phrases = get_indexed_phrases_per_example(
            phrases, tokenizer=tokenizer, vocab=vocab)
        indexed_chunks = get_indexed_phrases_per_example(
            chunks, tokenizer=tokenizer, vocab=vocab)
        # indexed_negative_chunks = get_indexed_phrases_per_example(
        #     chunks_negative, tokenizer=tokenizer, vocab=vocab)

        if debug:
            print(chunks)
            print(indexed_chunks)

        indexed_data = [indexed_phrases,
                        indexed_chunks]  # indexed_negative_chunks]

        n_examples = max(map(get_number_examples, indexed_data))
        max_length_examples = max(map(get_max_length_examples, indexed_data))
        max_length_phrases = max(map(get_max_length_phrases, indexed_data))

        padding_dim = (n_examples, max_length_examples, max_length_phrases)

        def __get_padded_examples(examples):
            return get_padded_examples(examples, padding_dim=padding_dim, padding_value=0)

        try:

            try:
                phrases, phrases_mask = __get_padded_examples(indexed_phrases)
            except ValueError:
                errors[0] += 1
                errors_f[0].append(f)

                print()
                print("ERROR: phrases")
                raise

            try:
                chunks, chunks_mask = __get_padded_examples(indexed_chunks)
            except ValueError:
                errors[1] += 1
                errors_f[1].append(f)
                print()
                print("ERROR: chunks")
                raise

            # try:
            #     chunks_negative, chunks_negative_mask = __get_padded_examples(
            #         indexed_negative_chunks)
            # except ValueError:
            #     print()
            #     print("ERROR: chunks_negative")
            #     raise

            try:
                sentence, sentence_mask = get_padded_examples(
                    indexed_sentences,
                    padding_value=0,
                    padding_dim=(get_number_examples(indexed_sentences),
                                 get_max_length_examples(indexed_sentences),
                                 get_max_length_phrases(indexed_sentences)))
            except ValueError:
                errors[2] += 1
                errors_f[2].append(f)
                print()
                print("ERROR: sentence")
                raise

            try:
                yago_entities, yago_entities_mask = get_padded_examples([
                    tokenization_ewiser_entities(el) for el in yago_entities__],
                    padding_value=0,
                    padding_dim=(n_examples, max_length_examples, 10)
                )
            except ValueError:
                errors[3] += 1
                errors_f[3].append(f)
                print()
                print("ERROR: yago_entities")
                raise

            try:
                phrases_2_crd, phrases_2_crd_mask = get_padded_examples(
                    phrases_2_crd,
                    padding_value=0,
                    padding_dim=(n_examples, max_length_examples, 4),
                    dtype=torch.float)
            except ValueError:
                errors[4] += 1
                errors_f[4].append(f)
                print()
                print("ERROR: phrases_2_crd")
                raise

        except ValueError:
            logging.error(f"ERROR: example i={i} on file {f}")

    ##
    print()

    print(f"errors: {errors}")
    print()

    for e in errors_f:
        print(e)
        print()
