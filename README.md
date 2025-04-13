# COMP7404_group1

# Conformer Speech Recognition (Hybrid CTC + Attention)

This repository contains an implementation of a hybrid speech recognition system based on the Conformer encoder and a Transformer-style decoder. The model is trained and evaluated on the LibriSpeech dataset (train-clean-100 subset).

Run preprocessing:

```
python scripts/preprocess_data.py \
  --train_dir path/to/train-clean-100 \
  --dev_dir path/to/dev-clean \
  --test_dir path/to/test-clean \
  --output_dir data
```

For training:
```
python train.py \
  --data_dir data \
  --epochs 35 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --mode joint \
  --ctc_weight 0.3
```
