import torch

if __name__ == "__main__":
    # num examples in batch = 1
    # num phrases max = 2
    # num words max = 5
    t = torch.Tensor([[[True, True, True, True, True], [True, True, True, False, False]]])

    print(t.size())

    s = torch.sum(t.type(torch.long), dim=-1)

    print(s)
    print(s.size())
    print(s.view(-1).size())
