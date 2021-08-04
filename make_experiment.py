import pyperclip
import datetime
import json


def decode_dict(s):
    import ast
    try:
        return ast.literal_eval(s)
    except:
        return {}


def diff(d1, d2):
    d = {}
    for k in d1.keys():
        if k in d2.keys():
            if d1[k] != d2[k]:
                d = {**d, k: d2[k]}
    return d


def quote(s):
    if isinstance(s, str):
        return '"{}"'.format(s)
    return s


default_params = {
    "mode": 0,
    "dataset": "flickr",
    "restore": None,
    "suffix": 'default',
    "develop": True,
    "device": "cpu",

    # dataloader
    "batch_size": 64,
    "num_workers": 0,
    "prefetch_factor": 2,
    "load_subset": None,
    "load_first": False,
    "load_first_img": False,

    # learning
    "learning_rate": 0.001,
    "grad_clipping": 1,
    "scheduler_gamma": 0.9,
    "n_epochs": 15,
    "align_loss": "kl-sem",
    "align_loss_kl_threshold": 0.5,
    "regression_loss": "iou_c-sem",
    "dropout_ratio": 0.3,  # not used for now
    'loss_weight_pred': 1,
    'loss_weight_reg': 1,
    'loss_weight_entities': 0.001,
    "n_active_box": 3,

    # network size
    "embeddings_text": "glove",
    "embeddings_freeze": True,
    "lstm_dim": 500,
    "lstm_num_layers": 1,
    "fusion_dim": 2053,
    "text_emb_size": 300,
    "yago_emb_size": 100,
    "yago_fusion_size": 300,
    "yago_n_entities": 2,
    "semantic_space_size": 500,
}

print("epxeriment number? [0]: ", end="")
e_no = input() or 0
e = f"e{e_no}"

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")
print(f"date? [{date}]: ", end="")
date = input() or date

print("commit? []: ", end="")
commit = input()
commit_visible = commit[:7]
commit_link = f"https://github.com/lparolari/VTKEL-solver/tree/{commit}"

print("dataset? [R]: ", end="")
dataset = input() or "R"

print("description? []: ", end="")
description = input()

print("note? []: ", end="")
note = input()

print("loss? [0]: ", end="")
loss = input() or 0
loss = float(loss)
loss = round(loss, 6)
loss = f"{loss:6f}"

print("epoch? [0]: ", end="")
loss_epoch = input() or 0

print("accuracy? [0]: ", end="")
accuracy = input() or 0
accuracy = float(accuracy)
accuracy = round(accuracy, 6)
accuracy = f"{accuracy:6f}"

print("epoch? [0]: ", end="")
accuracy_epoch = input() or 0

print("max_epoch? [15]: ", end="")
max_epoch = input() or 15

print("param (JSON)? []: ", end="")
params = input() or ""
params = decode_dict(params)
params = diff(default_params, params)
params = map(lambda x: f"`{x[0]}: {quote(x[1])}`", params.items())
params = ", ".join(params)

gen = f"| [{e}](#{date}-{e}) ({dataset}) | {date} | {loss} ({loss_epoch}/{max_epoch}) | {accuracy} ({accuracy_epoch}/{max_epoch}) | {params} | [{commit_visible}]({commit_link}) | {description} | {note} |"

print()
print(gen)
print()
print("Automatically copied to clupboard")

pyperclip.copy(gen)
