import torch


def collate_fn(batch):

    feats, tokens, texts = zip(*batch)
    batch_size = len(feats)

    input_lengths = [feat.shape[0] for feat in feats]
    target_lengths = [tok.shape[0] for tok in tokens]


    max_input_len = max(input_lengths)
    feat_dim = feats[0].shape[1]  # (n_mels)
    padded_feats = torch.zeros((batch_size, max_input_len, feat_dim), dtype=torch.float)
    for i, feat in enumerate(feats):
        T = feat.shape[0]
        padded_feats[i, :T, :] = feat

    max_target_len = max(target_lengths)

    padded_tokens = torch.zeros((batch_size, max_target_len), dtype=torch.long)
    for i, tok in enumerate(tokens):
        L = tok.shape[0]
        padded_tokens[i, :L] = tok

    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return padded_feats, padded_tokens, input_lengths, target_lengths, list(texts)
