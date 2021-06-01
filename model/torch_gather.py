"""
With this snippet we study how to retrieve the last relevant LSTM representation, which correspond to the neuron which
has seen the whole phrase.

In the example we have 2 phrases with max length of 5 words. However, the first phrase has 5 words, while the sencod
has 3 words. What we want to do is to retrieve the last relevant representation for both two phrases. In order to
do this, we use the [torch.gather](https://pytorch.org/docs/stable/generated/torch.gather.html) function, passing
the correct ids.
"""
import torch

if __name__ == "__main__":
    # [n_words_max, n_ph_max, repr_dim]
    # [5, 2, 3]

    # the LSTM output
    t = torch.Tensor([[[1, 2, 3], [101, 102, 103]],
                      [[4, 5, 6], [104, 105, 106]],
                      [[7, 8, 9], [107, 108, 109]],
                      [[10, 11, 12], [0, 0, 0]],
                      [[13, 14, 15], [0, 0, 0]]])

    # the ids corresponding to last relevant representation for each phrase
    idx = torch.Tensor([[[4, 4, 4], [2, 2, 2]]]).type(torch.long)

    print(t.size())    # [5, 2, 3]
    print(idx.size())  # [1, 2, 3]

    # gather that representation: we gather the components of the last relevant representation for each word for both
    # of both phrases
    x = torch.gather(t, 0, idx)

    print(x.size())    # [1, 2, 3]
    print(x)
    # tensor([[[ 13.,  14.,  15.],
    #          [107., 108., 109.]]])
