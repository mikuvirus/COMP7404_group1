import os
import argparse
import torch
import torchaudio
import sentencepiece as spm
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="LibriSpeech Data preprocessing")
    parser.add_argument('--train_dir', type=str, required=True,
                        help='train-clean-100 Unzip folder path')
    parser.add_argument('--dev_dir', type=str, default='',
                        help='dev-clean Unzip folder path')
    parser.add_argument('--test_dir', type=str, default='',
                        help='test-clean Unzip folder path')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output the directory of processed data')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_text_path = os.path.join(output_dir, "train_text.txt")
    with open(train_text_path, 'w', encoding='utf-8') as text_file:

        for root, _, files in os.walk(args.train_dir):
            for fname in files:
                if fname.endswith(".trans.txt"):
                    with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line == "":
                                continue

                            parts = line.split(" ", 1)
                            if len(parts) == 2:
                                _, transcript = parts
                                transcript = transcript.strip().lower()
                                text_file.write(transcript + "\n")

    print(">> train SentencePiece (vocab_size=1000, BPE)...")
    spm.SentencePieceTrainer.train(
        input=train_text_path,
        model_prefix=os.path.join(output_dir, "spm"),
        vocab_size=1000,
        model_type='bpe',
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1
    )
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(output_dir, "spm.model"))
    vocab_size = sp.get_piece_size()
    print(f">> vocab size: {vocab_size}")

    feat_dir = os.path.join(output_dir, "feats")
    os.makedirs(feat_dir, exist_ok=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    feat_train_dir = os.path.join(feat_dir, "train")
    os.makedirs(feat_train_dir, exist_ok=True)
    train_manifest_path = os.path.join(output_dir, "train.tsv")
    train_manifest = open(train_manifest_path, 'w', encoding='utf-8')

    for root, _, files in os.walk(args.train_dir):
        for fname in files:
            if fname.endswith(".flac"):
                audio_path = os.path.join(root, fname)
                utt_id = fname.replace(".flac", "")

                trans_filename = "-".join(utt_id.split("-")[:2]) + ".trans.txt"
                trans_file = os.path.join(root, trans_filename)
                transcript = ""
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith(utt_id + " "):
                            transcript = line.strip().split(" ", 1)[1]
                            break
                transcript = transcript.lower().strip()

                waveform, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)

                mel_spec = mel_transform(waveform)  # [1, n_mels, T]
                mel_spec = amp_to_db(mel_spec).squeeze(0)  # -> [n_mels, T]

                feat_path = os.path.join(feat_train_dir, f"{utt_id}.pt")
                torch.save(mel_spec, feat_path)

                token_ids = sp.encode(transcript, out_type=int)
                token_id_str = " ".join(map(str, token_ids))
                train_manifest.write(f"{os.path.abspath(feat_path)}\t{token_id_str}\t{transcript}\n")

    train_manifest.close()

    if args.dev_dir and os.path.isdir(args.dev_dir):

        feat_dev_dir = os.path.join(feat_dir, "dev")
        os.makedirs(feat_dev_dir, exist_ok=True)
        dev_manifest_path = os.path.join(output_dir, "dev.tsv")
        dev_manifest = open(dev_manifest_path, 'w', encoding='utf-8')

        for root, _, files in os.walk(args.dev_dir):
            for fname in files:
                if fname.endswith(".flac"):
                    audio_path = os.path.join(root, fname)
                    utt_id = fname.replace(".flac", "")
                    trans_filename = "-".join(utt_id.split("-")[:2]) + ".trans.txt"
                    trans_file = os.path.join(root, trans_filename)
                    transcript = ""
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(utt_id + " "):
                                transcript = line.strip().split(" ", 1)[1]
                                break
                    transcript = transcript.lower().strip()

                    waveform, sr = torchaudio.load(audio_path)
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    mel_spec = mel_transform(waveform)
                    mel_spec = amp_to_db(mel_spec).squeeze(0)

                    feat_path = os.path.join(feat_dev_dir, f"{utt_id}.pt")
                    torch.save(mel_spec, feat_path)

                    token_ids = sp.encode(transcript, out_type=int)
                    token_id_str = " ".join(map(str, token_ids))
                    dev_manifest.write(f"{os.path.abspath(feat_path)}\t{token_id_str}\t{transcript}\n")

        dev_manifest.close()

    if args.test_dir and os.path.isdir(args.test_dir):
        feat_test_dir = os.path.join(feat_dir, "test")
        os.makedirs(feat_test_dir, exist_ok=True)
        test_manifest_path = os.path.join(output_dir, "test.tsv")
        test_manifest = open(test_manifest_path, 'w', encoding='utf-8')

        for root, _, files in os.walk(args.test_dir):
            for fname in files:
                if fname.endswith(".flac"):
                    audio_path = os.path.join(root, fname)
                    utt_id = fname.replace(".flac", "")
                    trans_filename = "-".join(utt_id.split("-")[:2]) + ".trans.txt"
                    trans_file = os.path.join(root, trans_filename)
                    transcript = ""
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(utt_id + " "):
                                transcript = line.strip().split(" ", 1)[1]
                                break
                    transcript = transcript.lower().strip()

                    waveform, sr = torchaudio.load(audio_path)
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    mel_spec = mel_transform(waveform)
                    mel_spec = amp_to_db(mel_spec).squeeze(0)

                    feat_path = os.path.join(feat_test_dir, f"{utt_id}.pt")
                    torch.save(mel_spec, feat_path)

                    token_ids = sp.encode(transcript, out_type=int)
                    token_id_str = " ".join(map(str, token_ids))
                    test_manifest.write(f"{os.path.abspath(feat_path)}\t{token_id_str}\t{transcript}\n")

        test_manifest.close()

    print(">> complete")


if __name__ == "__main__":
    main()
