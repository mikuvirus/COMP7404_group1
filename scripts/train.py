import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data_utils import AudioDataset
from utils.collate_fn import collate_fn
from utils.tokenizer import SentencePieceTokenizer
from utils.metrics import calculate_wer, calculate_cer
from models.conformer import Conformer
from models.conformer_ctc import ConformerCTCModel
from models.conformer_hybrid import ConformerHybridModel

# 参数解析
parser = argparse.ArgumentParser(description="Conformer ASR 模型训练")
parser.add_argument('--data_dir', type=str, default='data', help='预处理后数据所在目录（包含spm.model和tsv文件）')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Adam优化器初始学习率')
parser.add_argument('--num_workers', type=int, default=2, help='DataLoader的工作进程数')
parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA/GPU')
parser.add_argument('--no_amp', action='store_true', help='禁用混合精度训练')
parser.add_argument('--mode', type=str, choices=['ctc', 'attn', 'joint'], default='joint',
                    help="Training mode: 'ctc' for CTC loss only, 'attn' for attention (seq2seq) loss only, 'joint' for combined loss.")
parser.add_argument('--ctc_weight', type=float, default=0.3,
                    help="Weight for CTC loss in joint mode (e.g., 0.3 means 30% CTC + 70% attention).")
parser.add_argument('--debug', action='store_true', help="是否开启调试模式，只跑 1 个 batch 验证代码正确性")
parser.add_argument('--resume_path', type=str, default=None, help="加载预训练模型或断点路径")


args = parser.parse_args()

use_cuda = (not args.no_cuda) and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

tokenizer = SentencePieceTokenizer(os.path.join(args.data_dir, "spm.model"))
vocab_size = tokenizer.vocab_size()
blank_id = vocab_size
num_classes = vocab_size + 1

train_manifest = os.path.join(args.data_dir, "train.tsv")
dev_manifest = os.path.join(args.data_dir, "dev.tsv")
test_manifest = os.path.join(args.data_dir, "test.tsv")

train_dataset = AudioDataset(train_manifest)
dev_dataset = AudioDataset(dev_manifest) if os.path.isfile(dev_manifest) else None
test_dataset = AudioDataset(test_manifest) if os.path.isfile(test_manifest) else None

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=args.num_workers)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers) if dev_dataset else None
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=args.num_workers) if test_dataset else None


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

if args.resume_path:
    ckpt = torch.load(args.resume_path, map_location=device)
    model.load_state_dict(ckpt)

model.to(device)

criterion_ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)
criterion_attn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding in target
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
use_amp = (not args.no_amp) and use_cuda
scaler = torch.amp.GradScaler(device_type='cuda') if use_amp else None

writer = SummaryWriter(log_dir=os.path.join(args.data_dir, "logs"))

best_wer = float('inf')
best_epoch = 0
best_model_path = os.path.join(args.data_dir, "best_model.pth")



def main():

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        max_steps = 1 if args.debug else float('inf')
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        step = 0
        for batch in progress_bar:
            padded_feats, padded_tokens, input_lengths, target_lengths, _ = batch
            padded_feats = padded_feats.to(device)
            padded_tokens = padded_tokens.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    if args.mode == 'ctc':
                        # CTC-only forward (no target needed)
                        logits = model(padded_feats, mode='ctc')  # [B, T_enc, num_classes]
                        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T_enc, B, num_classes]
                        loss = criterion_ctc(log_probs, padded_tokens, input_lengths.cpu(), target_lengths.cpu())
                    elif args.mode == 'attn':
                        # Attention-only forward (provide targets for teacher forcing)
                        attn_logits = model(padded_feats, padded_tokens,
                                            input_lengths=input_lengths, target_lengths=target_lengths,
                                            mode='attn')  # [B, T_out, num_classes]
                        # Compute cross-entropy loss over sequence, ignoring pad=0
                        loss = criterion_attn(attn_logits.view(-1, num_classes),
                                              padded_tokens.view(-1))
                    elif args.mode == 'joint':
                        # Joint mode: get both outputs
                        logits_ctc, attn_logits = model(padded_feats, padded_tokens,
                                                        input_lengths=input_lengths, target_lengths=target_lengths,
                                                        mode='joint')
                        # CTC loss on encoder output
                        log_probs = F.log_softmax(logits_ctc, dim=-1).transpose(0, 1)  # [T_enc, B, num_classes]
                        loss_ctc = criterion_ctc(log_probs, padded_tokens, input_lengths.cpu(), target_lengths.cpu())
                        # Attention loss on decoder output
                        loss_attn = criterion_attn(attn_logits.view(-1, num_classes),
                                                   padded_tokens.view(-1))
                        # Weighted sum of losses
                        loss = args.ctc_weight * loss_ctc + (1 - args.ctc_weight) * loss_attn
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()



            else:
                if args.mode == 'ctc':
                    logits = model(padded_feats, mode='ctc')
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    loss = criterion_ctc(log_probs, padded_tokens, input_lengths.cpu(), target_lengths.cpu())
                elif args.mode == 'attn':
                    attn_logits = model(padded_feats, padded_tokens,
                                        input_lengths=input_lengths, target_lengths=target_lengths,
                                        mode='attn')
                    loss = criterion_attn(attn_logits.view(-1, num_classes), padded_tokens.view(-1))
                elif args.mode == 'joint':
                    logits_ctc, attn_logits = model(padded_feats, padded_tokens,
                                                    input_lengths=input_lengths, target_lengths=target_lengths,
                                                    mode='joint')
                    log_probs = F.log_softmax(logits_ctc, dim=-1).transpose(0, 1)
                    loss_ctc = criterion_ctc(log_probs, padded_tokens, input_lengths.cpu(), target_lengths.cpu())
                    loss_attn = criterion_attn(attn_logits.view(-1, num_classes), padded_tokens.view(-1))
                    loss = args.ctc_weight * loss_ctc + (1 - args.ctc_weight) * loss_attn
                loss.backward()
                optimizer.step()

            step += 1
            if step >= max_steps:
                break
            if step % 500 == 0 and args.mode != 'attn':
                model.eval()
                with torch.no_grad():
                    sample_feats = padded_feats[0:1]  # [1, T, 80]
                    outputs = model(sample_feats)
                    if isinstance(outputs, tuple):
                        logits, output_lengths = outputs
                    else:
                        logits = outputs
                        output_lengths = input_lengths[0:1]

                    pred_ids = logits.argmax(dim=-1)[0].cpu().numpy().tolist()
                    length = int(output_lengths[0].cpu().item()) if isinstance(output_lengths, torch.Tensor) else len(
                        pred_ids)
                    pred_ids = pred_ids[:length]

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

                    hypothesis_text = tokenizer.ids_to_text(hyp_tokens) if len(hyp_tokens) > 0 else ""

                    print("\n[训练中预测 @ step {}]".format(step))
                    print("预测 token ids:", hyp_tokens)
                    print("预测文本:", hypothesis_text)
                    print("-" * 40)
                model.train()

            batch_size = padded_feats.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if dev_loader:
            model.eval()
            all_references = []
            all_hypotheses = []
            with torch.no_grad():
                for batch in dev_loader:
                    padded_feats, padded_tokens, input_lengths, target_lengths, texts = batch
                    padded_feats = padded_feats.to(device)

                    if args.mode == 'ctc':
                        logits = model(padded_feats, mode='ctc')
                        output_lengths = input_lengths
                    elif args.mode == 'attn':
                        logits = model(padded_feats, padded_tokens,
                                       input_lengths=input_lengths, target_lengths=target_lengths,
                                       mode='attn')
                        output_lengths = target_lengths
                    else:  # joint
                        logits, _ = model(padded_feats, padded_tokens,
                                          input_lengths=input_lengths, target_lengths=target_lengths,
                                          mode='joint')
                        output_lengths = input_lengths

                    pred_ids_batch = logits.argmax(dim=-1)  # [batch, T]

                    pred_ids_batch = pred_ids_batch.cpu().numpy()

                    out_lengths = output_lengths.cpu().numpy() if isinstance(output_lengths,
                                                                             torch.Tensor) else input_lengths.numpy()


                    for i in range(pred_ids_batch.shape[0]):

                        length = int(out_lengths[i])

                        if args.mode == 'attn':

                            pred_ids = pred_ids_batch[i][:length]
                            hyp_tokens = pred_ids.tolist() if isinstance(pred_ids, torch.Tensor) else list(pred_ids)

                        else:

                            pred_ids = pred_ids_batch[i][:length]
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

                        hypothesis_text = tokenizer.ids_to_text(hyp_tokens) if hyp_tokens else ""
                        all_hypotheses.append(hypothesis_text)
                    all_references.extend(texts)

            wer_score = calculate_wer(all_references, all_hypotheses)
            cer_score = calculate_cer(all_references, all_hypotheses)
            writer.add_scalar("Metrics/WER", wer_score, epoch)
            writer.add_scalar("Metrics/CER", cer_score, epoch)
            print(f"Epoch {epoch} - 验证集 WER: {wer_score * 100:.2f}%, CER: {cer_score * 100:.2f}%")

            if wer_score < best_wer:
                best_wer = wer_score
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
        else:

            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch


    if test_loader:

        best_model = ConformerHybridModel(
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
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.to(device)
        best_model.eval()
        all_references = []
        all_hypotheses = []
        with torch.no_grad():
            for batch in test_loader:
                padded_feats, padded_tokens, input_lengths, target_lengths, texts = batch
                padded_feats = padded_feats.to(device)
                outputs = best_model(padded_feats)
                if isinstance(outputs, tuple):
                    logits, output_lengths = outputs
                else:
                    logits = outputs
                    output_lengths = input_lengths
                pred_ids_batch = logits.argmax(dim=-1).cpu().numpy()
                out_lengths = output_lengths.cpu().numpy() if isinstance(output_lengths,
                                                                         torch.Tensor) else input_lengths.numpy()
                for i in range(pred_ids_batch.shape[0]):
                    length = int(out_lengths[i])
                    pred_ids = pred_ids_batch[i][:length]
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
                    if len(hyp_tokens) == 0:
                        hypothesis_text = ""
                    else:
                        hypothesis_text = tokenizer.ids_to_text(hyp_tokens)
                    all_hypotheses.append(hypothesis_text)
                all_references.extend(texts)
        if len(all_references) == 0:
            wer_score = 1.0
            cer_score = 1.0
        else:
            wer_score = calculate_wer(all_references, all_hypotheses)
            cer_score = calculate_cer(all_references, all_hypotheses)
            print(f"测试集评估 - WER: {wer_score * 100:.2f}%, CER: {cer_score * 100:.2f}%")

    print(f"训练结束，最佳模型出现在第 {best_epoch} 轮，已保存至 {best_model_path}")
    writer.close()

if __name__ == '__main__':
    main()