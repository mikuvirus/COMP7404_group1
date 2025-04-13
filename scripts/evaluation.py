import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.data_utils import AudioDataset
from utils.collate_fn import collate_fn
from utils.tokenizer import SentencePieceTokenizer
from utils.metrics import calculate_wer, calculate_cer
from models.conformer_hybrid import ConformerHybridModel


def evaluate_model(model, data_loader, tokenizer, blank_id, device, mode):
    model.eval()
    all_hypotheses = []
    all_references = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
            feats, tokens, input_lengths, target_lengths, texts = batch
            feats = feats.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            if mode == 'ctc':
                logits = model(feats, mode='ctc')
                output_lengths = input_lengths
            elif mode == 'attn':
                logits = model(feats, tokens, input_lengths, target_lengths, mode='attn')
                output_lengths = target_lengths
            else:
                logits, _ = model(feats, tokens, input_lengths, target_lengths, mode='joint')
                output_lengths = input_lengths

            pred_ids_batch = logits.argmax(dim=-1).cpu().numpy()
            out_lengths = output_lengths.cpu().numpy()

            for i in range(pred_ids_batch.shape[0]):
                length = int(out_lengths[i])
                pred_ids = pred_ids_batch[i][:length]

                if mode == 'attn':
                    hyp_tokens = list(pred_ids)
                else:
                    hyp_tokens = []
                    prev_token = -1
                    for token_id in pred_ids:
                        if token_id == blank_id:
                            prev_token = -1
                            continue
                        if token_id == prev_token:
                            continue
                        hyp_tokens.append(token_id)
                        prev_token = token_id

                hyp_text = tokenizer.ids_to_text(hyp_tokens) if hyp_tokens else ""
                all_hypotheses.append(hyp_text)
                all_references.append(texts[i])

    wer = calculate_wer(all_references, all_hypotheses)
    cer = calculate_cer(all_references, all_hypotheses)
    return wer, cer


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER of long/short validation sets")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--short_manifest', type=str, required=True)
    parser.add_argument('--long_manifest', type=str, required=True)
    parser.add_argument('--sp_model', type=str, required=True)
    parser.add_argument('--mode', type=str, default='joint', choices=['ctc', 'attn', 'joint'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    tokenizer = SentencePieceTokenizer(args.sp_model)
    vocab_size = tokenizer.vocab_size()
    blank_id = vocab_size
    num_classes = vocab_size + 1

    model = ConformerHybridModel(
        input_dim=80,
        num_classes=num_classes,
        encoder_dim=144,
        depth=12,
        dim_head=64,
        heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        decoder_layers=6,
        decoder_heads=8
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    short_dataset = AudioDataset(args.short_manifest)
    long_dataset = AudioDataset(args.long_manifest)
    short_loader = DataLoader(short_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    long_loader = DataLoader(long_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    print("Start evaluating the short voice validation set...")
    short_wer, short_cer = evaluate_model(model, short_loader, tokenizer, blank_id, device, args.mode)
    print(f"short validation sets - WER: {short_wer * 100:.2f}%, CER: {short_cer * 100:.2f}%")

    print("Start evaluating the long voice validation set...")
    long_wer, long_cer = evaluate_model(model, long_loader, tokenizer, blank_id, device, args.mode)
    print(f"long validation sets - WER: {long_wer * 100:.2f}%, CER: {long_cer * 100:.2f}%")


if __name__ == '__main__':
    main()
