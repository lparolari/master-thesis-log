import sys

filename = sys.argv[1]

# read csv
with open(filename, "r") as f:
    lines = f.readlines()

    for l in lines:
        l = l.replace("\n", "")

        lx = l.split(",")
        ly = lx

        if len(lx) > 3:
            ly = ["/".join(lx[:-2]), lx[-2], lx[-1]]

        print(",".join(ly))
