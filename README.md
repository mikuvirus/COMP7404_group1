# COMP7404_group1

# Conformer Speech Recognition (Hybrid CTC + Attention)

This repository contains an implementation of a hybrid speech recognition system based on the Conformer encoder and a Transformer-style decoder. The model is trained and evaluated on the LibriSpeech dataset (train-clean-100 subset).

Install dependencies:
```
pip install -r requirements.txt
```



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
For Evaluation


Split dev.tsv into long and short sets:
```
python scripts/split_by_duration.py --manifest data/dev.tsv --output_dir data_split
```

Evaluate:
```
python evaluation.py \
  --model_path data/best_model.pth \
  --short_manifest data_split/dev_short.tsv \
  --long_manifest data_split/dev_long.tsv \
  --sp_model data/spm.model \
  --mode joint
```




