import torch
from torch import nn
from torch.nn import functional as F

from models.pytorch.mlp import MLP

__AGGREGATION_MODES = ["add", "concat"]


class AttentionInteraction(nn.Module):
    def __init__(self, dim_embedding, num_layers, num_heads=1, dropout=0.0, **attn_kwargs):
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                (
                    nn.MultiheadAttention(
                        embed_dim=dim_embedding,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        **attn_kwargs,
                    ),
                    nn.LayerNorm(dim_embedding),
                )
            )

    def forward(self, inputs):
        outputs = inputs
        for mha, layer_norm in self.layers:
            outputs = layer_norm(mha(outputs, outputs, outputs) + outputs)  # (bs, num_embedding, dim_embedding)

        return outputs


class AutoInt(nn.Module):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_attention=1,
        num_heads=1,
        num_hidden=2,
        dim_hidden=16,
        dropout=0.0,
        aggregation_mode="add",
    ):
        super().__init__()

        if aggregation_mode not in __AGGREGATION_MODES:
            raise ValueError(f"'aggregation_mode' must be one of {__AGGREGATION_MODES}")

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.aggregation_mode = aggregation_mode

        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=dim_embedding)

        self.interaction_mlp = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=1 if aggregation_mode == "add" else None,
            dropout=dropout,
        )

        self.interaction_attn = AttentionInteraction(
            dim_embedding=dim_embedding,
            num_layers=num_attention,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.projection_head = nn.Linear(dim_input * dim_embedding, 1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # (bs, num_emb, dim_emb)

        attn_out = self.interaction_attn(embeddings)  # (bs, num_emb, dim_emb)
        attn_out = torch.reshape(attn_out, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        mlp_out = torch.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)
        mlp_out = self.interaction_mlp(mlp_out)

        if self.aggregation_mode == "add":
            attn_out = self.projection_head(attn_out)  # (bs, 1)
            logits = attn_out + mlp_out  # (bs, 1)
        elif self.aggregation_mode == "concat":
            latent = torch.concat((attn_out, mlp_out), dim=-1)  # (bs, dim_input * dim_emb + dim_hidden)
            logits = self.projection_head(latent)  # (bs, 1)

        outputs = F.sigmoid(logits)  # (bs, 1)

        return outputs
