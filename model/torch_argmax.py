import torch

if __name__ == "__main__":
    t1 = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
    t2 = torch.Tensor([[[1, 2, 3], [7, 8, 9]]])

    i = torch.argmax(t, dim=-1, keepdim=False)

    print(i.size())
    print(i)

    t1 = torch.argmax(t1, dim=-1)
    t2 = torch.argmax(t2, dim=-1)

    print(torch.sum((t1 - t2 == 0).type(torch.long)))
