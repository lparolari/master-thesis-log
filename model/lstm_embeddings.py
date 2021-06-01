import collections
import json
import torch
import torchtext
from torch.nn.utils import rnn
import torch.nn.functional as F

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
    phrase111 = "Two young guys with shaggy hair look at their hands while hanging out in the yard ."
    phrase112 = "Two young , White males are outside near many bushes ."
    phrase121 = "Several men in hard hats are operating a giant pulley system ."
    example11 = [phrase111, phrase112]
    example12 = [phrase121]
    batch1 = [example11, example12]

    text_embeddings = create_embeddings_network()
    text_representation = create_representation_network()

    # get phrases tensor

    # [b, n_ph_max, n_words_max]
    phrases, phrases_mask = get_phrases_tensor(batch1)
    print(f"phrases_mask.size() = {phrases_mask.size()}")

    # get phrases length
    # [b, n_ph_max]
    phrases_length = torch.sum(phrases_mask.type(torch.long), dim=-1)
    print(f"phrases_length.size() = {phrases_length.size()}")

    # get embeddings and manipulate them for RNN
    # [b, n_ph_max, n_words_max, emb_dim]
    phrases_embeddings = text_embeddings(phrases)
    print(f"phrases_embeddings.size() = {phrases_embeddings.size()}")
    # [b*n_ph_max, n_words_max, emb_dim]
    phrases_embeddings = phrases_embeddings.view(-1, phrases_embeddings.size()[-2], phrases_embeddings.size()[-1])
    print(f"phrases_embeddings.size() = {phrases_embeddings.size()}")
    # [n_words_max, b*n_ph_max, emb_dim]
    phrases_embeddings = phrases_embeddings.permute(1, 0, 2)
    phrases_embeddings = phrases_embeddings.contiguous()  # `.contiguous()` returns a tensor contiguous in memory
    print(f"phrases_embeddings.size() = {phrases_embeddings.size()}")

    # [b*n_ph_max]
    phrases_length_clamp = phrases_length.view(-1).clamp(min=1).cpu()
    print(f"phrases_length_clamp.size() = {phrases_length_clamp.size()}")

    # [ *** internal representation *** ]
    phrases_pack_embeddings = rnn.pack_padded_sequence(phrases_embeddings, phrases_length_clamp, enforce_sorted=False)

    # [n_words_max, b*n_ph_max, repr_dim], ([1, b*n_ph_max, repr_dim], [1, b*n_ph_max, repr_dim])
    #   where the 1 is the result of num_layers * num_directions of the LSTM (1, 1 in our case)
    phrases_x_output, (phrases_x_hidden, phrases_x_cell) = text_representation(phrases_pack_embeddings)

    # Get back our padded sequence. First component of the tuple is a tensor with representation from LSTM,
    # while the second is a tensor representing the real length of phrases
    # [n_words_max, b*n_ph_max, repr_dim], [b*n_ph_max]
    phrases_x_output, phrases_x_output_length = rnn.pad_packed_sequence(phrases_x_output, batch_first=False)

    print(f"phrases_x_output.size() = {phrases_x_output.size()}")
    print(f"phrases_x_output_length.size() = {phrases_x_output_length.size()}")
    print(f"phrases_x_hidden.size() = {phrases_x_hidden.size()}")
    print(f"phrases_x_cell.size() = {phrases_x_cell.size()}")

    # representation is the last (valid) neuron of the phrase in the LSTM, we need gather ids of these neurons
    # for then collecting them

    idx = (phrases_x_output_length - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, phrases_x_output.size()[2])
    print(f"idx.size() = {idx.size()}")

    # gather phrases features
    phrases_x = torch.gather(phrases_x_output, 0, idx)
    print(f"phrases_x.size() = {phrases_x.size()}")
    phrases_x = phrases_x.permute(1, 0, 2).contiguous()
    print(f"phrases_x.size() = {phrases_x.size()}")
    phrases_x = phrases_x.unsqueeze(0)
    print(f"phrases_x.size() = {phrases_x.size()}")
    phrases_x = phrases_x.view(phrases_mask.size()[0], phrases_mask.size()[1], 500)
    print(f"phrases_x.size() = {phrases_x.size()}")

    print(f"phrases_mask.size() = {phrases_mask.size()}")

    # test whether there is at least one True in phrases_mask, i.e., the phrase is not synthetic. Note that we may
    # get synthetic phrases when, e.g., the number of phrases per example differs.
    # [b, n_phrases, 1]
    phrases_synthetic_mask = torch.any(phrases_mask, dim=-1, keepdim=True).type(torch.long)
    print(f"phrases_synthetic_mask.size() = {phrases_synthetic_mask.size()}")

    # if we have a synthetic phrase we set to 0 its representation
    # [b, n_phrases, repr_dim]
    phrases_x = torch.masked_fill(phrases_x, phrases_synthetic_mask == 0, 0)
    print(f"phrases_x.size() = {phrases_x.size()}")

    # [b, n_phrases, repr_dim]
    phrases_x_norm = F.normalize(phrases_x, p=1, dim=-1)

    print(f"phrases_x_norm.size() = {phrases_x_norm.size()}")
    print(phrases_x_norm)
