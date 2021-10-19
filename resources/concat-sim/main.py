import torch


def main():
    # TODO

    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    a = a / torch.norm(a, p=2)

    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    b = b / torch.norm(b, p=2)

    c = torch.tensor([7, 8, 9], dtype=torch.float32)
    c = c / torch.norm(c, p=2)

    d = torch.tensor([10, 11, 12], dtype=torch.float32)
    d = d / torch.norm(d, p=2)

    ac = torch.cat((a, c))
    ac = ac / torch.norm(ac, p=2)

    ab = torch.cat((a, b))
    ab = ab / torch.norm(ab, p=2)

    bd = torch.cat((b, d))
    ab = ab / torch.norm(bd, p=2)

    cd = torch.cat((c, d))
    cd = cd / torch.norm(cd, p=2)

    sim_a_b = torch.cosine_similarity(a, b, dim=-1)
    sim_c_d = torch.cosine_similarity(c, d, dim=-1)
    sim_ac_bd = torch.cosine_similarity(ac, bd, dim=-1)
    sim_ab_cd = torch.cosine_similarity(ab, cd, dim=-1)

    print("sim(a, b) + sim(c, d) = sim(ac, bd) ==>",
          torch.equal(sim_a_b + sim_c_d, sim_ac_bd))
    print("sim(a, b) + sim(c, d) = sim(ab, cd) ==>",
          torch.equal(sim_a_b + sim_c_d, sim_ab_cd))


if __name__ == "__main__":
    main()
