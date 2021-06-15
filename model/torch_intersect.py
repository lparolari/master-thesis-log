import torch

if __name__ == "__main__":
    def torch_intersect(t1, t2, dim=None):
        u1, u2 = t1.unique(dim=dim), t2.unique(dim=dim)
        c = torch.cat((u1, u2))
        uniques, counts = c.unique(return_counts=True, dim=dim)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]
        return intersection, difference


    t1 = torch.Tensor([[1, 1, 2, 2], [2, 2, 3, 3]])
    t2 = torch.Tensor([[3, 3, 4, 5], [4, 4, 5, 5]])

    print(torch_intersect(t1, t2, dim=1))
