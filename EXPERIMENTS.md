# Experiments

<!-- prettier-ignore-start -->

| Id | Date | Loss (Epoch)\* | Accuracy (Epoch)\* | Params\*\* | Commit | Desc | Note |
| -- | ---- | ------------ | ---------------- | ------ | ------ | ---- | ---- |
| [#4](#2021-07-06-4) | 2021-07-06 | -12.14 (5) | 8.49 (3) | `batch_size: 128` | [ddb1d42](https://github.com/lparolari/VTKEL-solver/commit/ddb1d4226e51b7d017f2836a993887e51b631503) | repulsion with `n_falsy: 1` on full dataset | training accuracy always increasing up to 5% in 10 epochs, interrupted at epoch 11 |
| [#3](#2021-07-05-3) | 2021-07-05 | -12.70 (9) | 3.86 (7) | `load_subset: 0.2, batch_size: 128` | [ddb1d42](https://github.com/lparolari/VTKEL-solver/commit/ddb1d4226e51b7d017f2836a993887e51b631503) | repulsion with `n_falsy: 1` | |
| [#2](#2021-07-05-2) | 2021-07-05 | - | - | `load_subset: 0.2, batch_size: 128` | [20c765e](https://github.com/lparolari/VTKEL-solver/commit/20c765e483f0906d3718aa178293172573802644) | training with fix on arloss (minus) | interrupted |
| [#1](#2021-07-03-1) | 2021-07-03 | -49.30 (8) | 4.40 (15) | `batch_size: 128` | [d4232b1](https://github.com/lparolari/VTKEL-solver/commit/d4232b1720deaad7c1f8ceb3f2ce6f02795c6017) | fisrt complete training on referit | |

<!-- prettier-ignore-end -->

\* on validation dataset \
\*\* diff from defaults

## Blob

### 2021-07-06 (#4)

<details>
<summary>History</summary>

```
Model started with the following parameters:
{'mode': 0, 'dataset': 'referit', 'restore': None, 'suffix': 'default', 'develop': True, 'device': 'cuda', 'batch_size': 128, 'num_workers': 1, 'prefetch_factor': 1, 'load_subset': None, 'load_first': False, 'load_first_img': False, 'learning_rate': 0.001, 'grad_clipping': 1, 'scheduler_gamma': 0.9, 'n_epochs': 15, 'align_loss': 'kl-sem', 'align_loss_kl_threshold': 0.5, 'regression_loss': 'iou_c-sem', 'dropout_ratio': 0.3, 'loss_weight_pred': 1, 'loss_weight_reg': 1, 'loss_weight_entities': 0.001, 'n_falsy': 1, 'embeddings_text': 'glove', 'embeddings_freeze': True, 'lstm_dim': 500, 'lstm_num_layers': 1, 'fusion_dim': 2053, 'text_emb_size': 300, 'yago_emb_size': 100, 'yago_fusion_size': 300, 'yago_n_entities': 2, 'semantic_space_size': 500, 'folder_img': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/refer', 'folder_results': '/home/2/2019/lparolar/Downloads/results/referit', 'folder_data': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/preprocessed', 'folder_idx_train': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/train.txt', 'folder_idx_valid': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/val.txt', 'folder_idx_test': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/test.txt'}
Loading training dataset.
Loading validation dataset.
Loading test dataset.
------------- START MODEL TRAINING
----- Epoch: 1
--- Training completed.   Loss: -11.735872, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.2620, PAccuracy: 27.6339 .
--- Validation completed.   Loss: -11.792890, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4544, PAccuracy: 28.2171 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_1.pth ..
Epoch 1 completed in 3 hours, 39 minute and 11 seconds .
----- Epoch: 2
--- Training completed.   Loss: -11.811697, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3560, PAccuracy: 27.1088 .
--- Validation completed.   Loss: -11.835632, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.8328, PAccuracy: 28.8882 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_2.pth ..
Epoch 2 completed in 3 hours, 38 minute and 1 seconds .
----- Epoch: 3
--- Training completed.   Loss: -11.804035, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4972, PAccuracy: 27.2806 .
--- Validation completed.   Loss: -11.853614, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.9673, PAccuracy: 28.5338 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_3.pth ..
Epoch 3 completed in 3 hours, 31 minute and 2 seconds .
----- Epoch: 4
--- Training completed.   Loss: -11.755250, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.8699, PAccuracy: 27.3449 .
--- Validation completed.   Loss: -11.751869, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 8.4917, PAccuracy: 30.5327 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_4.pth ..
Epoch 4 completed in 3 hours, 46 minute and 41 seconds .
----- Epoch: 5
--- Training completed.   Loss: -11.832404, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.2407, PAccuracy: 27.2563 .
--- Validation completed.   Loss: -12.140813, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.5195, PAccuracy: 29.2715 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_5.pth ..
Epoch 5 completed in 3 hours, 29 minute and 15 seconds .
----- Epoch: 6
--- Training completed.   Loss: -11.770277, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.2416, PAccuracy: 27.2632 .
--- Validation completed.   Loss: -11.893083, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.8566, PAccuracy: 29.6137 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_6.pth ..
Epoch 6 completed in 3 hours, 23 minute and 31 seconds .
----- Epoch: 7
--- Training completed.   Loss: -11.803903, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.3486, PAccuracy: 27.4181 .
--- Validation completed.   Loss: -11.861039, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.7095, PAccuracy: 28.2261 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_7.pth ..
Epoch 7 completed in 3 hours, 19 minute and 55 seconds .
----- Epoch: 8
--- Training completed.   Loss: -11.793687, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.4903, PAccuracy: 27.4453 .
--- Validation completed.   Loss: -11.758370, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.4435, PAccuracy: 28.4439 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_8.pth ..
Epoch 8 completed in 3 hours, 15 minute and 36 seconds .
----- Epoch: 9
--- Training completed.   Loss: -11.785093, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.8921, PAccuracy: 27.6607 .
--- Validation completed.   Loss: -11.937345, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.7427, PAccuracy: 28.4224 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_9.pth ..
Epoch 9 completed in 3 hours, 16 minute and 23 seconds .
----- Epoch: 10
--- Training completed.   Loss: -11.833655, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 5.1284, PAccuracy: 27.8996 .
--- Validation completed.   Loss: -11.751447, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 5.9845, PAccuracy: 30.4609 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_10.pth ..
Epoch 10 completed in 3 hours, 20 minute and 51 seconds .
----- Epoch: 11
--- Training completed.   Loss: -11.825187, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 5.0749, PAccuracy: 28.0697 .
--- Validation completed.   Loss: -11.803118, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.9495, PAccuracy: 28.6756 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_11.pth ..
Epoch 11 completed in 3 hours, 19 minute and 28 seconds .
```

</details>

### 2021-07-05 (#3)

<details>
<summary>History</summary>

```
Model started with the following parameters:
{'mode': 0, 'dataset': 'referit', 'restore': None, 'suffix': 'default', 'develop': True, 'device': 'cuda', 'batch_size': 128, 'num_workers': 1, 'prefetch_factor': 1, 'load_subset': 0.2, 'load_first': False, 'load_first_img': False, 'learning_rate': 0.001, 'grad_clipping': 1, 'scheduler_gamma': 0.9, 'n_epochs': 15, 'align_loss': 'kl-sem', 'align_loss_kl_threshold': 0.5, 'regression_loss': 'iou_c-sem', 'dropout_ratio': 0.3, 'loss_weight_pred': 1, 'loss_weight_reg': 1, 'loss_weight_entities': 0.001, 'n_falsy': 1, 'embeddings_text': 'glove', 'embeddings_freeze': True, 'lstm_dim': 500, 'lstm_num_layers': 1, 'fusion_dim': 2053, 'text_emb_size': 300, 'yago_emb_size': 100, 'yago_fusion_size': 300, 'yago_n_entities': 2, 'semantic_space_size': 500, 'folder_img': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/refer', 'folder_results': '/home/2/2019/lparolar/Downloads/results/referit', 'folder_data': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/preprocessed', 'folder_idx_train': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/train.txt', 'folder_idx_valid': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/val.txt', 'folder_idx_test': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/test.txt'}
Loading training dataset.
Loading validation dataset.
Loading test dataset.
------------- START MODEL TRAINING
----- Epoch: 1
--- Training completed.   Loss: -11.644444, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3212, PAccuracy: 28.6821 .
--- Validation completed.   Loss: -12.171728, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.3825, PAccuracy: 27.0077 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_1.pth ..
Epoch 1 completed in 0 hours, 37 minute and 52 seconds .
----- Epoch: 2
--- Training completed.   Loss: -12.060690, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.1479, PAccuracy: 28.1038 .
--- Validation completed.   Loss: -12.173510, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.5552, PAccuracy: 26.0955 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_2.pth ..
Epoch 2 completed in 0 hours, 21 minute and 43 seconds .
----- Epoch: 3
--- Training completed.   Loss: -12.069280, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.2750, PAccuracy: 27.5218 .
--- Validation completed.   Loss: -12.306858, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.1397, PAccuracy: 26.3473 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_3.pth ..
Epoch 3 completed in 0 hours, 14 minute and 52 seconds .
----- Epoch: 4
--- Training completed.   Loss: -12.007876, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.2839, PAccuracy: 26.9632 .
--- Validation completed.   Loss: -12.320462, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.2028, PAccuracy: 26.9375 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_4.pth ..
Epoch 4 completed in 0 hours, 14 minute and 56 seconds .
----- Epoch: 5
--- Training completed.   Loss: -11.936259, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3518, PAccuracy: 27.1059 .
--- Validation completed.   Loss: -12.514828, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.0575, PAccuracy: 26.5309 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_5.pth ..
Epoch 5 completed in 0 hours, 14 minute and 51 seconds .
----- Epoch: 6
--- Training completed.   Loss: -11.980361, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3911, PAccuracy: 27.0232 .
--- Validation completed.   Loss: -11.879200, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.9668, PAccuracy: 27.0274 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_6.pth ..
Epoch 6 completed in 0 hours, 14 minute and 50 seconds .
----- Epoch: 7
--- Training completed.   Loss: -11.924987, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3986, PAccuracy: 27.0807 .
--- Validation completed.   Loss: -12.279340, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.8564, PAccuracy: 28.1745 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_7.pth ..
Epoch 7 completed in 0 hours, 14 minute and 44 seconds .
----- Epoch: 8
--- Training completed.   Loss: -11.965241, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5600, PAccuracy: 27.1239 .
--- Validation completed.   Loss: -12.537053, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.6278, PAccuracy: 27.2560 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_8.pth ..
Epoch 8 completed in 0 hours, 14 minute and 45 seconds .
----- Epoch: 9
--- Training completed.   Loss: -11.855393, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3997, PAccuracy: 27.0774 .
--- Validation completed.   Loss: -12.699197, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3818, PAccuracy: 27.2776 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_9.pth ..
Epoch 9 completed in 0 hours, 14 minute and 46 seconds .
----- Epoch: 10
--- Training completed.   Loss: -11.989465, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3533, PAccuracy: 27.3355 .
--- Validation completed.   Loss: -12.357894, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.4939, PAccuracy: 27.1330 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_10.pth ..
Epoch 10 completed in 0 hours, 14 minute and 44 seconds .
----- Epoch: 11
--- Training completed.   Loss: -11.975657, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3913, PAccuracy: 27.1633 .
--- Validation completed.   Loss: -12.372195, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.2377, PAccuracy: 27.4988 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_11.pth ..
Epoch 11 completed in 0 hours, 14 minute and 41 seconds .
----- Epoch: 12
--- Training completed.   Loss: -11.925183, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4667, PAccuracy: 27.2263 .
--- Validation completed.   Loss: -12.120575, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.1361, PAccuracy: 27.1050 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_12.pth ..
Epoch 12 completed in 0 hours, 15 minute and 33 seconds .
----- Epoch: 13
--- Training completed.   Loss: -12.027928, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5101, PAccuracy: 27.3155 .
--- Validation completed.   Loss: -12.389876, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4708, PAccuracy: 27.3364 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_13.pth ..
Epoch 13 completed in 0 hours, 16 minute and 4 seconds .
----- Epoch: 14
--- Training completed.   Loss: -11.860335, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.6235, PAccuracy: 27.0173 .
--- Validation completed.   Loss: -12.595021, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4622, PAccuracy: 27.7551 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_14.pth ..
Epoch 14 completed in 0 hours, 16 minute and 4 seconds .
----- Epoch: 15
--- Training completed.   Loss: -11.851944, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5186, PAccuracy: 27.1721 .
--- Validation completed.   Loss: -12.607153, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.6319, PAccuracy: 27.5655 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_15.pth ..
Epoch 15 completed in 0 hours, 16 minute and 6 seconds .
Best hist loss at epoch 9: -12.699196697316399 .
Best hist accuracy at epoch 7: 3.856386132420709 .
Model training end.
```

</details>

### 2021-07-05 (#2)

<details>
<summary>History</summary>

```
Model started with the following parameters:
{'mode': 0, 'dataset': 'referit', 'restore': None, 'suffix': 'default', 'develop': True, 'device': 'cuda', 'batch_size': 128, 'num_workers': 1, 'prefetch_factor': 1, 'load_subset': 0.2, 'load_first': False, 'load_first_img': False, 'learning_rate': 0.001, 'grad_clipping': 1, 'scheduler_gamma': 0.9, 'n_epochs': 15, 'align_loss': 'kl-sem', 'align_loss_kl_threshold': 0.5, 'regression_loss': 'iou_c-sem', 'dropout_ratio': 0.3, 'loss_weight_pred': 1, 'loss_weight_reg': 1, 'loss_weight_entities': 0.001, 'embeddings_text': 'glove', 'embeddings_freeze': True, 'lstm_dim': 500, 'lstm_num_layers': 1, 'fusion_dim': 2053, 'text_emb_size': 300, 'yago_emb_size': 100, 'yago_fusion_size': 300, 'yago_n_entities': 2, 'semantic_space_size': 500, 'folder_img': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/refer', 'folder_results': '/home/2/2019/lparolar/Downloads/results/referit', 'folder_data': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/preprocessed', 'folder_idx_train': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/train.txt', 'folder_idx_valid': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/val.txt', 'folder_idx_test': '/aulahomes2/2/2019/lparolar/Thesis/VTKEL-solver/data/referit_raw/test.txt'}
Loading training dataset.
Loading validation dataset.
Loading test dataset.
------------- START MODEL TRAINING
----- Epoch: 1
--- Training completed.   Loss: -48.067662, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5143, PAccuracy: 28.4248 .
--- Validation completed.   Loss: -49.007339, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.3297, PAccuracy: 27.8404 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_1.pth ..
Epoch 1 completed in 0 hours, 41 minute and 50 seconds .
----- Epoch: 2
--- Training completed.   Loss: -49.226255, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.2467, PAccuracy: 26.5938 .
--- Validation completed.   Loss: -49.075031, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.0659, PAccuracy: 28.3562 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_2.pth ..
Epoch 2 completed in 0 hours, 20 minute and 28 seconds .
----- Epoch: 3
--- Training completed.   Loss: -49.229109, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3282, PAccuracy: 26.8148 .
--- Validation completed.   Loss: -49.233624, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.3277, PAccuracy: 28.5719 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_3.pth ..
Epoch 3 completed in 0 hours, 17 minute and 39 seconds .
----- Epoch: 4
--- Training completed.   Loss: -49.186241, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.1121, PAccuracy: 26.8968 .
--- Validation completed.   Loss: -49.226857, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.8217, PAccuracy: 28.3879 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_4.pth ..
Epoch 4 completed in 0 hours, 17 minute and 38 seconds .
----- Epoch: 5
--- Training completed.   Loss: -49.288240, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4767, PAccuracy: 26.9451 .
--- Validation completed.   Loss: -49.279651, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.9931, PAccuracy: 28.3651 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_5.pth ..
Epoch 5 completed in 0 hours, 18 minute and 26 seconds .
----- Epoch: 6
--- Training completed.   Loss: -49.265880, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5014, PAccuracy: 26.9830 .
--- Validation completed.   Loss: -49.233782, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.8996, PAccuracy: 29.9078 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_6.pth ..
Epoch 6 completed in 0 hours, 18 minute and 50 seconds .
----- Epoch: 7
--- Training completed.   Loss: -49.285719, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5694, PAccuracy: 26.8883 .
--- Validation completed.   Loss: -49.074438, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.8295, PAccuracy: 28.9710 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_7.pth ..
Epoch 7 completed in 0 hours, 15 minute and 41 seconds .
----- Epoch: 8
--- Training completed.   Loss: -49.339065, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.5995, PAccuracy: 27.2227 .
--- Validation completed.   Loss: -49.128896, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.9157, PAccuracy: 28.5492 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_8.pth ..
Epoch 8 completed in 0 hours, 15 minute and 3 seconds .
----- Epoch: 9
--- Training completed.   Loss: -49.254838, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4781, PAccuracy: 27.2797 .
--- Validation completed.   Loss: -49.283275, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.0882, PAccuracy: 30.2863 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_9.pth ..
Epoch 9 completed in 0 hours, 15 minute and 28 seconds .
----- Epoch: 10
--- Training completed.   Loss: -49.193615, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.6342, PAccuracy: 27.0327 .
--- Validation completed.   Loss: -49.122438, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.7667, PAccuracy: 30.8546 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_10.pth ..
Epoch 10 completed in 0 hours, 15 minute and 24 seconds .
----- Epoch: 11
--- Training completed.   Loss: -49.190422, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4101, PAccuracy: 27.1396 .
--- Validation completed.   Loss: -49.447351, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.0730, PAccuracy: 29.9056 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_11.pth ..
Epoch 11 completed in 0 hours, 14 minute and 57 seconds .
----- Epoch: 12
--- Training completed.   Loss: -49.189930, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.7478, PAccuracy: 27.2097 .
--- Validation completed.   Loss: -49.008951, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.7497, PAccuracy: 29.4369 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_12.pth ..
Epoch 12 completed in 0 hours, 14 minute and 55 seconds .
```

</details>

### 2021-07-03 (#1)

<details>
<summary>History</summary>

```
--- Training completed.   Loss: -47.651709, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.7689, PAccuracy: 21.8205 .
--- Validation completed.   Loss: -49.172170, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4175, PAccuracy: 24.7158 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_1.pth ..
Epoch 1 completed in 3 hours, 10 minute and 50 seconds .
----- Epoch: 2
--- Training completed.   Loss: -49.237166, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.9098, PAccuracy: 22.4612 .
--- Validation completed.   Loss: -49.234307, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.4112, PAccuracy: 24.9071 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_2.pth ..
Epoch 2 completed in 3 hours, 6 minute and 13 seconds .
----- Epoch: 3
--- Training completed.   Loss: -49.233987, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.3591, PAccuracy: 23.1055 .
--- Validation completed.   Loss: -49.223755, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.9126, PAccuracy: 24.4438 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_3.pth ..
Epoch 3 completed in 3 hours, 5 minute and 20 seconds .
----- Epoch: 4
--- Training completed.   Loss: -49.215859, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.8484, PAccuracy: 23.4422 .
--- Validation completed.   Loss: -49.204719, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.5475, PAccuracy: 25.3256 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_4.pth ..
Epoch 4 completed in 3 hours, 7 minute and 11 seconds .
----- Epoch: 5
--- Training completed.   Loss: -49.195249, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.7520, PAccuracy: 23.5436 .
--- Validation completed.   Loss: -49.146962, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.4307, PAccuracy: 24.7494 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_5.pth ..
Epoch 5 completed in 2 hours, 58 minute and 22 seconds .
----- Epoch: 6
--- Training completed.   Loss: -49.228855, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.6019, PAccuracy: 23.7248 .
--- Validation completed.   Loss: -49.185350, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.8653, PAccuracy: 24.8980 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_6.pth ..
Epoch 6 completed in 2 hours, 54 minute and 55 seconds .
----- Epoch: 7
--- Training completed.   Loss: -49.225569, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.4979, PAccuracy: 23.9620 .
--- Validation completed.   Loss: -49.116095, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.5492, PAccuracy: 24.6120 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_7.pth ..
Epoch 7 completed in 2 hours, 50 minute and 30 seconds .
----- Epoch: 8
--- Training completed.   Loss: -49.232501, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.4546, PAccuracy: 23.8514 .
--- Validation completed.   Loss: -49.298435, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.7622, PAccuracy: 25.1897 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_8.pth ..
Epoch 8 completed in 2 hours, 45 minute and 27 seconds .
----- Epoch: 9
--- Training completed.   Loss: -49.227131, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.3388, PAccuracy: 23.8306 .
--- Validation completed.   Loss: -49.126582, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.4991, PAccuracy: 25.0845 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_9.pth ..
Epoch 9 completed in 2 hours, 45 minute and 23 seconds .
----- Epoch: 10
--- Training completed.   Loss: -49.233788, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.5124, PAccuracy: 23.7752 .
--- Validation completed.   Loss: -49.269709, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.3150, PAccuracy: 25.6379 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_10.pth ..
Epoch 10 completed in 2 hours, 45 minute and 6 seconds .
----- Epoch: 11
--- Training completed.   Loss: -49.217475, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 1.8250, PAccuracy: 24.1116 .
--- Validation completed.   Loss: -49.177079, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.2564, PAccuracy: 25.7158 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_11.pth ..
Epoch 11 completed in 2 hours, 46 minute and 37 seconds .
----- Epoch: 12
--- Training completed.   Loss: -49.244046, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.4225, PAccuracy: 24.7755 .
--- Validation completed.   Loss: -49.266846, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.6129, PAccuracy: 25.8289 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_12.pth ..
Epoch 12 completed in 2 hours, 45 minute and 12 seconds .
----- Epoch: 13
--- Training completed.   Loss: -49.185037, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.6638, PAccuracy: 25.3318 .
--- Validation completed.   Loss: -49.108692, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.0558, PAccuracy: 26.1473 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_13.pth ..
Epoch 13 completed in 2 hours, 46 minute and 3 seconds .
----- Epoch: 14
--- Training completed.   Loss: -49.214966, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.8971, PAccuracy: 25.7201 .
--- Validation completed.   Loss: -49.153296, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 2.8308, PAccuracy: 27.0187 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_14.pth ..
Epoch 14 completed in 2 hours, 45 minute and 30 seconds .
----- Epoch: 15
--- Training completed.   Loss: -49.229144, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 3.1440, PAccuracy: 25.8246 .
--- Validation completed.   Loss: -49.065015, Reg_loss: 0.000000, Pred_loss: 0.000000, Accuracy: 4.4050, PAccuracy: 27.3909 .
Saved model: /home/2/2019/lparolar/Downloads/results/referit/model_default_15.pth ..
Epoch 15 completed in 2 hours, 46 minute and 4 seconds .
Best hist loss at epoch 8: -49.29843537140736 .
Best hist accuracy at epoch 15: 4.404956110724148 .
Model training end.
```

</details>

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
