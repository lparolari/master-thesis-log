import torch

if __name__ == "__main__":
    # b, n_phrases, 4 -> gt bb for each phrase
    t = torch.Tensor([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 0, 1], [2, 3, 4, 5]]])
    print(t.size())

    # b, n_chunks -> index of query matching chunk
    i = torch.Tensor([[1, 0], [0, 0]]).unsqueeze(-1).repeat(1, 1, 4).long()
    print(i.size())

    # output: [b, n_chunks, 4] the bounding box gathered from phrase with index matching chunk

    print(torch.gather(t, 1, i))
