import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conformer import Conformer  # reuse the existing Conformer encoder

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the decoder tokens."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)  # not a parameter, but saved for use

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        L = x.size(1)
        # Add positional encoding up to sequence length L
        return x + self.pe[:, :L]

class ConformerHybridModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        encoder_dim: int = 144,
        depth: int = 12,
        dim_head: int = 64,
        heads: int = 4,
        ff_mult: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff_dim: int = None
    ):
        """
        Hybrid Conformer model with CTC and Attention.
        :param input_dim: Dimension of input features (e.g., 80 for FBank).
        :param num_classes: Vocabulary size *including* CTC blank token.
        :param encoder_dim: Dimension of Conformer encoder output.
        :param depth: Number of Conformer encoder layers.
        :param dim_head: Dimension per attention head in Conformer.
        :param heads: Number of attention heads in Conformer.
        :param ff_mult: Feed-forward expansion factor in Conformer.
        :param conv_expansion_factor: Conformer conv module expansion factor.
        :param conv_kernel_size: Kernel size for Conformer conv module.
        :param decoder_layers: Number of Transformer decoder layers.
        :param decoder_heads: Number of heads in Transformer decoder.
        :param decoder_ff_dim: Dimensionality of decoder feedforward layers (default 4*decoder_dim).
        """
        super().__init__()
        decoder_dim = encoder_dim  # Use the same dimension for decoder model
        if decoder_ff_dim is None:
            decoder_ff_dim = decoder_dim * 4

        # 1. Input linear to project mel features to model dimension
        self.input_linear = nn.Linear(input_dim, encoder_dim)
        # 2. Conformer encoder (same as existing, for acoustic modeling)
        self.encoder = Conformer(dim=encoder_dim, depth=depth,
                                 dim_head=dim_head, heads=heads,
                                 ff_mult=ff_mult,
                                 conv_expansion_factor=conv_expansion_factor,
                                 conv_kernel_size=conv_kernel_size)
        # 3. CTC output layer (linear projection from encoder hidden dim to vocab size)
        self.ctc_output = nn.Linear(encoder_dim, num_classes)
        # 4. Transformer decoder components for attention-based modeling
        self.token_embedding = nn.Embedding(num_classes, decoder_dim)
        self.pos_encoding = PositionalEncoding(decoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=decoder_heads,
                                                  dim_feedforward=decoder_ff_dim, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.attn_output = nn.Linear(decoder_dim, num_classes)  # output projection for decoder

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None,
                input_lengths: torch.Tensor = None, target_lengths: torch.Tensor = None,
                mode: str = 'ctc'):
        """
        Forward pass for the hybrid model.
        :param x: Acoustic input features [B, T_enc, input_dim].
        :param targets: Target token sequences [B, T_out] (padded with 0) for teacher-forcing (needed for attn/joint).
        :param input_lengths: Actual lengths of each input sequence (optional, for masking).
        :param target_lengths: Actual lengths of each target sequence (optional, for masking).
        :param mode: 'ctc', 'attn', or 'joint'.
        """
        # Encode acoustic features with Conformer encoder
        # Project input to encoder dimension:
        x_proj = self.input_linear(x)                   # [B, T_enc, encoder_dim]
        enc_out = self.encoder(x_proj)                  # [B, T_enc, encoder_dim]
        # If CTC mode, simply return encoder outputs after CTC linear layer
        if mode == 'ctc':
            logits_ctc = self.ctc_output(enc_out)       # [B, T_enc, num_classes]
            return logits_ctc  # (Note: CTC loss will be computed outside)
        # For attention or joint modes, we require target sequences for teacher forcing
        if targets is None:
            raise ValueError("targets must be provided for attn or joint mode training")
        # Prepare decoder input by shifting targets right and adding BOS token at start
        B, T_out = targets.shape
        # Define BOS token as the CTC blank (last index in vocab) for convenience
        bos_token_id = self.ctc_output.out_features - 1  # blank index = num_classes - 1
        # Construct decoder input: [BOS] + targets[:,:-1]
        bos_col = torch.full((B, 1), bos_token_id, dtype=torch.long, device=targets.device)
        # Remove the last time-step of each target sequence (shift right)
        targets_shifted = torch.cat([bos_col, targets[:, :-1]], dim=1)  # shape [B, T_out]
        targets_shifted = targets_shifted.to(enc_out.device)  # 保证 device 一致
        # Create token embeddings and add positional encoding
        dec_in_emb = self.token_embedding(targets_shifted)             # [B, T_out, decoder_dim]
        dec_in_emb = self.pos_encoding(dec_in_emb)                     # [B, T_out, decoder_dim] with positional info
        # Transformer expects [T, B, D] shape for inputs, so transpose
        dec_in_emb = dec_in_emb.transpose(0, 1)                        # [T_out, B, decoder_dim]
        enc_memory = enc_out.transpose(0, 1)                           # [T_enc, B, encoder_dim] for decoder cross-attn
        # Prepare masks for the decoder
        T_dec = targets_shifted.size(1)  # length of decoder input sequence (should equal T_out)
        # 1) Causal mask for decoder self-attention (prevent attending to future tokens)
        # Using boolean mask: True = mask out (no attention), shape [T_dec, T_dec]
        tgt_mask = torch.triu(torch.ones(T_dec, T_dec, device=enc_out.device, dtype=torch.bool), diagonal=1)
        # 2) Padding mask for target sequence (mask out padding positions in decoder attention)
        tgt_key_padding_mask = None
        if target_lengths is not None:
            # Calculate effective decoder input lengths: for each sample, if it has target_length L,
            # decoder input includes BOS + (L-1) targets (unless L is max, where we dropped one token as BOS).
            max_T = T_out
            # If target_length == max_T (sequence is at max length), we dropped its last token when shifting.
            # Otherwise, decoder input length = target_length + 1.
            # Clamp to ensure we don't exceed max_T:
            dec_input_lengths = torch.clamp(target_lengths + 1, max=max_T).to(enc_out.device)
            # Create mask: True for positions beyond the actual decoder input length
            idx = torch.arange(max_T, device=enc_out.device)[None, :]  # shape [1, max_T]
            tgt_key_padding_mask = idx >= dec_input_lengths.unsqueeze(1)  # [B, max_T] bool
        # 3) Padding mask for encoder memory (mask out padded frames in encoder outputs)
        memory_key_padding_mask = None
        if input_lengths is not None:
            T_enc = enc_out.size(1)
            idx_enc = torch.arange(T_enc, device=enc_out.device)[None, :]  # [1, T_enc]
            input_lengths = input_lengths.to(enc_out.device)
            memory_key_padding_mask = idx_enc >= input_lengths.unsqueeze(1)  # [B, T_enc] bool
        # Run Transformer decoder
        decoder_output = self.decoder(dec_in_emb, enc_memory,
                                      tgt_mask=tgt_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        decoder_output = decoder_output.transpose(0, 1)  # [B, T_out, decoder_dim]
        attn_logits = self.attn_output(decoder_output)    # [B, T_out, num_classes]
        if mode == 'attn':
            return attn_logits
        elif mode == 'joint':
            # Compute both outputs
            logits_ctc = self.ctc_output(enc_out)         # [B, T_enc, num_classes]
            return logits_ctc, attn_logits
        else:
            raise ValueError(f"Unsupported mode: {mode}")
