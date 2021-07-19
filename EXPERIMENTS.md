# Experiments

<!-- prettier-ignore-start -->

<!-- 

TEMPLATE
========

| [eN](#yyyy-MM-dd-eN) (R/F) | yyyy-MM-dd | loss (epoch) | acc (epoch) | params | [commit](url) | desc | note |

-->

| Id (DS\*) | Date | Loss (Epoch)\*\* | Accuracy (Epoch)\*\* | Params\*\*\* | Commit | Desc | Note |
| -- | ---- | -------------- | ------------------ | ---------- | ------ | ---- | ---- |
| [e11](#2021-07-18-e11) (R) | 2021-07-18 | -0.319394 (6) | 5.9834 (2) | `n_falsy: 3`, `n_truthy: 3`, `suffix: "main-fix-43"` | [a207c71](https://github.com/lparolari/VTKEL-solver/tree/a207c719abdd1f39a36b3c880a1adc9ff6746313) | train on referit after fixing [#43](https://github.com/lparolari/VTKEL-solver/issues/43), i.e., fix wrong masking on iou | interruped at epoch 9 doe to bad performace |
| [e10](#2021-07-15-e10) (R) | 2021-07-15 | -0.316901 (6) | 7.5425 (1) | `n_falsy: 3`, `n_truthy: 3`, `suffix: "feat-no-semantic-embedding-for-chunks"` | [f3a8d1a](https://github.com/lparolari/VTKEL-solver/commit/f3a8d1a732fac0f9155b9f4cfde334e7cb2147b2) | directly use lstm output for chunks without embedding | interrupted at epoch 7 due to connection issues between working machine and data storage. accuracy gets worse from epoch to epoch but it's very smooth and values are usually better wrt other trainings |
| [e9](#2021-07-14-e9) (R) | 2021-07-14 | -0.318937 (8) | 7.6239 (2) | `n_falsy: 3`, `n_truthy: 3`, `suffix: "basic-random-k-3"` | [d4bb063](https://github.com/lparolari/VTKEL-solver/tree/d4bb0634bd639fa803606a3b54fb70240ecc60fb) | use random strategy for picking k-similar example to attract or repulse, fix negative example forward | interrupted at epoch 9 due to its weakness: no learning during training neither validation |
| [e8](#2021-07-13-e8) (R) | 2021-07-13 | -0.194748 (7) | 6.9326 (2) | `lstm_dim: 2053`, `n_falsy: 3`, `n_truthy: 3`, `suffix: "feat-no-semantic-embedding"` | [b754c52](https://github.com/lparolari/VTKEL-solver/tree/b754c52dd8a298c346acf0846691d45fe4e4123b) | train without semantic embedding for images, i.e., lstm_dim matches visual features size | interrupted due to no interesting results: accuracy in training phase went down |
| [e7](#2021-07-13-e7) (R) | 2021-07-13 | - | - | `batch_size: 128`, `n_falsy: 3`, `n_truthy: 3`, `get_similar_positive: "random"`, `get_similar_negative: "random"`, `suffix: "d202107131008"`, `restore: "model_tmp_1.pth"` | [d4bb063](https://github.com/lparolari/VTKEL-solver/commit/d4bb0634bd639fa803606a3b54fb70240ecc60fb) (real training done on [06f91b0](https://github.com/lparolari/VTKEL-solver/commit/06f91b0f8d74d25d7e370ccb92fe91637c2036c0) which has logs) | training with verbose output (scores tensor, min, max, average) after fixing negative example forward. changed the strategy for picking k-similar example to attract or repuls: now we use randomized strategy with k=3 | FAILED (see [e9](#2021-07-14-9)) |
| [e6](#2021-07-09-e6) (R) | 2021-07-09 | -0.038570 (3) | 10.0191 (2) | `batch_size: 128, n_falsy: 3, n_truthy: 3` | [48f28a1](https://github.com/lparolari/VTKEL-solver/commit/48f28a1cb1bdbdc6fcd39fa28722346b482b9a9d) | using similarity between bounding box classes and concepts from chunks | very long training (> 5h 30m) on referit using similarity between concepts and classes, crash at epoch 4 due to index error (see [#35]()) |
| [e5](#2021-07-08-e5) (R) | 2021-07-08 | -0.080238 (5) | 10.0901 (1) | `batch_size: 128, n_falsy: 3, n_truthy: 3` | [d4232b1](https://github.com/lparolari/VTKEL-solver/commit/bfcceb0d46390310b2fc090fc8b0f4cc62b07a03) | training on referit with no similarity and 3 truthy/falsy in loss | interrupted due to connectivity issues at epoch 6  |
| [e4](#2021-07-06-e4) (R) | 2021-07-06 | -12.14 (5) | 8.49 (3) | `batch_size: 128` | [bfcceb0](https://github.com/lparolari/VTKEL-solver/commit/ddb1d4226e51b7d017f2836a993887e51b631503) | repulsion with `n_falsy: 1` on full dataset | training accuracy always increasing up to 5% in 10 epochs, interrupted at epoch 11 |
| [e3](#2021-07-05-e3) (R) | 2021-07-05 | -12.70 (9) | 3.86 (7) | `load_subset: 0.2, batch_size: 128` | [ddb1d42](https://github.com/lparolari/VTKEL-solver/commit/ddb1d4226e51b7d017f2836a993887e51b631503) | repulsion with `n_falsy: 1` | |
| [e2](#2021-07-05-e2) (R) | 2021-07-05 | - | - | `load_subset: 0.2, batch_size: 128` | [20c765e](https://github.com/lparolari/VTKEL-solver/commit/20c765e483f0906d3718aa178293172573802644) | training with fix on arloss (minus) | interrupted |
| [e1](#2021-07-03-e1) (R) | 2021-07-03 | -49.30 (8) | 4.40 (15) | `batch_size: 128` | [d4232b1](https://github.com/lparolari/VTKEL-solver/commit/d4232b1720deaad7c1f8ceb3f2ce6f02795c6017) | fisrt complete training on referit | |

<!-- prettier-ignore-end -->

\* dataset, R=referit, F=flickr, C=mscoco\
\*\* on validation dataset \
\*\*\* diff from defaults

## Blob

<!--

TEMPLATE
========

### yyyy-MM-dd (eN)

History: [\[Readable\]](experiments/eN.history.txt)
[\[Full\]](experiments/eN.history.full.txt)

-->

### 2021-07-15 (e10)

History: [\[Readable\]](experiments/e10.history.txt)
[\[Full\]](experiments/e10.history.full.txt)

### 2021-07-14 (e9)

History: [\[Readable\]](experiments/e9.history.txt)
[\[Full\]](experiments/e9.history.full.txt)

### 2121-07-13 (e8)

History: [\[Readable\]](experiments/e8.history.txt)

### 2021-07-13 (e7)

History: missing data

### 2021-07-09 (e6)

History: [\[Readable\]](experiments/e6.history.txt)

### 2021-07-08 (e5)

History: [\[Readable\]](experiments/e5.history.txt)

### 2021-07-06 (e4)

History: [\[Readable\]](experiments/e4.history.txt)

### 2021-07-05 (e3)

History: [\[Readable\]](experiments/e3.history.txt)

### 2021-07-05 (e2)

History: [\[Readable\]](experiments/e2.history.txt)

### 2021-07-03 (e1)

History: [\[Readable\]](experiments/e1.history.txt)

<details>
<summary>Parameters</summary>

```js
{
    "mode": 0,
    "dataset": "referit",
    "restore": "None",
    "suffix": "default",
    "develop": true,
    "device": "cuda",
    "batch_size": 128,
    "num_workers": 1,
    "prefetch_factor": 1,
    "load_subset": "None",
    "load_first": false,
    "load_first_img": false,
    "learning_rate": 0.001,
    "grad_clipping": 1,
    "scheduler_gamma": 0.9,
    "n_epochs": 15,
    "align_loss": "kl-sem",
    "align_loss_kl_threshold": 0.5,
    "regression_loss": "iou_c-sem",
    "dropout_ratio": 0.3,
    "loss_weight_pred": 1,
    "loss_weight_reg": 1,
    "loss_weight_entities": 0.001,
    "embeddings_text": "glove",
    "embeddings_freeze": true,
    "lstm_dim": 500,
    "lstm_num_layers": 1,
    "fusion_dim": 2053,
    "text_emb_size": 300,
    "yago_emb_size": 100,
    "yago_fusion_size": 300,
    "yago_n_entities": 2,
    "semantic_space_size": 500,
    "folder_img": "/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/refer",
    "folder_results": "/home/2/2019/lparolar/Downloads/results/referit",
    "folder_data": "/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/preprocessed",
    "folder_idx_train": "/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/train.txt",
    "folder_idx_valid": "/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/val.txt",
    "folder_idx_test": "/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/test.txt"
}
```

</details>
