import torch

if __name__ == "__main__":
    # num examples in batch = 1
    # num phrases max = 2
    # num words max = 5
    t = torch.Tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]])

    print(t.size())

    s = torch.nn.Softmax(dim=2)(t)

    s = torch.sum(s, dim=1).squeeze()
    print(s)
    print(s.size())

    s = torch.sum(s, dim=0)

    print(s)
    print(s.size())