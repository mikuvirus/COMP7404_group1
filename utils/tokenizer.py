import torch
import sentencepiece as spm
import numpy as np


class SentencePieceTokenizer:

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def text_to_ids(self, text: str):
        return self.sp.encode(text, out_type=int)

    def ids_to_text(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if not isinstance(ids, list) or len(ids) == 0:
            return ""
        ids = [int(i) for i in ids]
        return self.sp.decode(ids)

    def vocab_size(self):
        return self.sp.get_piece_size()

    def id_to_token(self, id: int):
        return self.sp.id_to_piece(id)
