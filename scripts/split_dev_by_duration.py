import os
import argparse
import torchaudio
import torch
import sentencepiece as spm
from tqdm import tqdm

def split_dev_by_duration(dev_dir, output_dir, sp_model_path, max_duration=5.0):
    os.makedirs(output_dir, exist_ok=True)
    feat_dir = os.path.join(output_dir, "feats_dev")
    os.makedirs(feat_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    short_manifest = open(os.path.join(output_dir, "dev_short.tsv"), 'w', encoding='utf-8')
    long_manifest = open(os.path.join(output_dir, "dev_long.tsv"), 'w', encoding='utf-8')

    for root, _, files in os.walk(dev_dir):
        for fname in files:
            if not fname.endswith(".flac"):
                continue
            audio_path = os.path.join(root, fname)
            utt_id = fname.replace(".flac", "")
            trans_filename = "-".join(utt_id.split("-")[:2]) + ".trans.txt"
            trans_path = os.path.join(root, trans_filename)
            if not os.path.exists(trans_path):
                continue

            transcript = ""
            with open(trans_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(utt_id + " "):
                        transcript = line.strip().split(" ", 1)[1].lower()
                        break
            if transcript == "":
                continue

            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            duration = waveform.shape[1] / 16000

            mel_spec = mel_transform(waveform)
            mel_spec = amp_to_db(mel_spec).squeeze(0)

            feat_path = os.path.join(feat_dir, f"{utt_id}.pt")
            torch.save(mel_spec, feat_path)

            token_ids = sp.encode(transcript, out_type=int)
            token_id_str = " ".join(map(str, token_ids))
            manifest_line = f"{os.path.abspath(feat_path)}\t{token_id_str}\t{transcript}\n"

            if duration <= max_duration:
                short_manifest.write(manifest_line)
            else:
                long_manifest.write(manifest_line)

    short_manifest.close()
    long_manifest.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Divide the validation set into short and long sentences based on audio length")
    parser.add_argument('--dev_dir', type=str, required=True, help='dev-clean path')
    parser.add_argument('--output_dir', type=str, default='data_split', help='output path')
    parser.add_argument('--sp_model', type=str, required=True, help='spm.model path')
    parser.add_argument('--max_duration', type=float, default=5.0, help='Maximum duration of short voice (seconds)')
    args = parser.parse_args()

    split_dev_by_duration(args.dev_dir, args.output_dir, args.sp_model, args.max_duration)
