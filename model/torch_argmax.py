import torch

if __name__ == "__main__":
    t1 = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])

    print(f"t1.size() = {t1.size()}")  # torch.Size([1, 2, 3])

    t = torch.argmax(t1, dim=2, keepdim=False)

    print(f"t.size() = {t.size()}")  # torch.Size([1, 2])
    print(f"t = {t}")  # tensor([[2, 2]])
