# Experiments

<!-- prettier-ignore-start -->

| Id | Date | Loss (Epoch) | Accuracy (Epoch) | Params | Commit | Desc | Note |
| -- | ---- | ------------ | ---------------- | ------ | ------ | ---- | ---- |
| [#2](#2021-07-05---2-) | 2021-07-05 | - | - | `batch_size: 128` | [20c765e](https://github.com/lparolari/VTKEL-solver/commit/20c765e483f0906d3718aa178293172573802644) | training with fix on arloss (minus) | interrupted |
| [#1](#2021-07-03---1-) | 2021-07-03 | -49.29843537140736 (8) | 4.404956110724148 (15) | `batch_size: 128` | [d4232b1](https://github.com/lparolari/VTKEL-solver/commit/d4232b1720deaad7c1f8ceb3f2ce6f02795c6017) | fisrt complete training on referit | |

<!-- prettier-ignore-end -->

## Blob

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
