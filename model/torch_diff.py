import torch

if __name__ == "__main__":
    t1 = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
    t2 = torch.Tensor([[[1, 2, 3], [7, 8, 9]]])

    a1 = torch.argmax(t1, dim=-1)
    a2 = torch.argmax(t2, dim=-1)

    print(torch.sum((a1 - a2 == 0).type(torch.long)))
