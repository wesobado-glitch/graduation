import torch
import torch.nn as nn

from recommender.sasrec.domain.config import ModelConfig


class PointWiseFFN(nn.Module):
    """Two-layer point-wise feed-forward network (eq. 4 in SASRec paper)."""

    def __init__(self, d: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    LOCAL skip connections within each block:
      x = x_prev + attention(norm(x_prev))
      x = x_prev + ffn(norm(x_prev))
    """

    def __init__(
        self,
        num_items: int,
        max_len: int,
        cfg: ModelConfig,
    ):
        super().__init__()

        self.num_items = num_items
        self.d = cfg.d
        self.max_len = max_len
        self.num_blocks = cfg.num_blocks

        self.item_emb = nn.Embedding(num_items + 1, cfg.d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, cfg.d)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.attn_layer_norms = nn.ModuleList()
        self.ffn_layer_norms = nn.ModuleList()
        self.attn_dropouts = nn.ModuleList()
        self.ffn_dropouts = nn.ModuleList()

        for _ in range(cfg.num_blocks):
            self.attention_layers.append(
                nn.MultiheadAttention(cfg.d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            )
            self.ffn_layers.append(PointWiseFFN(cfg.d, cfg.dropout))
            self.attn_layer_norms.append(nn.LayerNorm(cfg.d))
            self.ffn_layer_norms.append(nn.LayerNorm(cfg.d))
            self.attn_dropouts.append(nn.Dropout(cfg.dropout))
            self.ffn_dropouts.append(nn.Dropout(cfg.dropout))

        self.last_layer_norm = nn.LayerNorm(cfg.d)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        input_seq : (batch, max_len)  — padded item indices
        Returns   : (batch, max_len, d)
        """
        batch_size, seq_len = input_seq.shape
        device = input_seq.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.item_emb(input_seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = (input_seq == 0).unsqueeze(-1)  # (B, L, 1)

        for i in range(self.num_blocks):
            residual = x
            x_norm = self.attn_layer_norms[i](x)
            attn_out, _ = self.attention_layers[i](
                x_norm, x_norm, x_norm,
                attn_mask=causal_mask,
            )
            attn_out = attn_out.masked_fill(pad_mask, 0.0)
            x = residual + self.attn_dropouts[i](attn_out)

            residual = x
            x_norm = self.ffn_layer_norms[i](x)
            ffn_out = self.ffn_layers[i](x_norm)
            ffn_out = ffn_out.masked_fill(pad_mask, 0.0)
            x = residual + self.ffn_dropouts[i](ffn_out)

        return self.last_layer_norm(x)

    def predict(self, input_seq: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        h = self.forward(input_seq)
        h_last = h[:, -1, :]
        item_embs = self.item_emb(item_indices)
        return torch.bmm(item_embs, h_last.unsqueeze(-1)).squeeze(-1)
