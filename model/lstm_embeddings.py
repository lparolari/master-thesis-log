import collections
import json
import torch
import torchtext

from text_tokenizer import get_phrases_tensor


vocab_path = "resources/vocab.json"
text_embeddings_size = 300
freeze = False


def create_representation_network():
    rnn = torch.nn.LSTM(input_size=300, hidden_size=500, num_layers=1, bidirectional=False, batch_first=False)

    # initialization
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

    return rnn


def create_embeddings_network():
    vocab_dict = load_json(vocab_path)
    vocab_counter = collections.Counter(vocab_dict)
    vocab = torchtext.vocab.Vocab(vocab_counter, specials=[
                                  '<pad>'], specials_first=True)

    glove_embeddings = torchtext.vocab.GloVe('840B', dim=300)

    return create_text_embeddings_matrix(vocab, glove_embeddings)


def create_text_embeddings_matrix(vocab, glove_embeddings):
    embedding_matrix_values = torch.zeros((len(vocab) + 1, text_embeddings_size), requires_grad=freeze)

    glove_words = glove_embeddings.stoi.keys()

    for idx in range(len(vocab)):
        word = vocab.itos[idx]

        # Use the GloVe embedding if word in vocabulary, otherwise we need to learn it
        if word in glove_words:
            glove_idx = glove_embeddings.stoi[word]
            embedding_matrix_values[idx, :] = glove_embeddings.vectors[glove_idx]
        else:
            torch.nn.init.normal_(embedding_matrix_values[idx, :])

    embedding_matrix = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=text_embeddings_size)
    embedding_matrix.weight = torch.nn.Parameter(embedding_matrix_values)
    embedding_matrix.weight.requires_grad = freeze

    return embedding_matrix


def load_json(file):
    with open(file, "r") as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    phrase1 = "Two young guys with shaggy hair look at their hands while hanging out in the yard ."
    phrase2 = "Two young , White males are outside near many bushes ."
    example1 = [phrase1, phrase2]
    batch1 = [example1]

    text_embeddings = create_embeddings_network()
    text_representation = create_representation_network()

    phrases, phrases_mask = get_phrases_tensor(batch1)
    phrases_length = torch.sum(phrases_mask.type(torch.long), dim=-1)

    phrases_embeddings = text_embeddings(phrases)
    phrases_embeddings = phrases_embeddings.view(-1, phrases_embeddings.size()[-2], phrases_embeddings.size()[-1])
    phrases_embeddings = phrases_embeddings.permute(1, 0, 2).contiguous()

    phrases_x_output, (phrases_x_hidden, phrases_x_cell) = text_representation(phrases_embeddings)

    print(f"phrases_x_output.size() = {phrases_x_output.size()}")
    print(f"phrases_x_hidden.size() = {phrases_x_hidden.size()}")
    print(f"phrases_x_cell.size() = {phrases_x_cell.size()}")
