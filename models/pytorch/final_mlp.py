import torch
from torch import nn
from torch.nn import functional as F

from models.pytorch.mlp import MLP


class FeatureSelection(nn.Module):
    def __init__(self, dim_input, num_hidden, dim_hidden, dropout):
        super().__init__()

        self.gate_1 = MLP(
            dim_in=dim_input,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_input,
            dropout=dropout,
            batch_norm=False,
        )

        self.gate_2 = MLP(
            dim_in=dim_input,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_input,
            dropout=dropout,
            batch_norm=False,
        )

    def forward(self, inputs):
        gate_score_1 = self.gate_1(inputs)  # (1, dim_input)
        out_1 = 2.0 * F.sigmoid(gate_score_1) * inputs  # (bs, dim_input)

        gate_score_2 = self.gate_2(inputs)  # (1, dim_input)
        out_2 = 2.0 * F.sigmoid(gate_score_2) * inputs  # (bs, dim_input)

        return out_1, out_2  # (bs, dim_input), (bs, dim_input)


class Aggregation(nn.Module):
    def __init__(self, dim_inputs_1, dim_inputs_2, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head_1 = dim_inputs_1 // num_heads
        self.dim_head_2 = dim_inputs_2 // num_heads

        self.w_1 = nn.Parameter(torch.empty(self.dim_head_1, num_heads, 1))
        self.w_2 = nn.Parameter(torch.empty(self.dim_head_2, num_heads, 1))
        self.w_12 = nn.Parameter(torch.empty(num_heads, self.dim_head_1, self.dim_head_2, 1))
        self.bias = nn.Parameter(torch.ones(1, num_heads, 1))

        self._reset_weights()

    def _reset_weights(self):
        nn.init.xavier_uniform_(self.w_1)
        nn.init.xavier_uniform_(self.w_2)
        nn.init.xavier_uniform_(self.w_12)

    def forward(self, inputs_1, inputs_2):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        inputs_1 = torch.reshape(inputs_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        inputs_2 = torch.reshape(inputs_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)

        first_order = torch.einsum("bhi,iho->bho", inputs_1, self.w_1)  # (bs, num_heads, 1)
        first_order += torch.einsum("bhi,iho->bho", inputs_2, self.w_2)  # (bs, num_heads, 1)
        second_order = torch.einsum("bhi,hijo,bhj->bho", inputs_1, self.w_12, inputs_2)  # (bs, num_heads, 1)

        out = torch.sum(first_order + second_order + self.bias, dim=1)  # (bs, 1)

        return out


class FinalMLP(nn.Module):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_hidden_fs,
        dim_hidden_fs,
        num_hidden_1,
        dim_hidden_1,
        num_hidden_2,
        dim_hidden_2,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=dim_embedding)

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_input=dim_input * dim_embedding,
            num_hidden=num_hidden_fs,
            dim_hidden=dim_hidden_fs,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dropout=dropout,
        )
        # branch 2
        self.interaction_2 = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dropout=dropout,
        )

        # final aggregation layer
        self.aggregation = Aggregation(
            dim_inputs_1=dim_hidden_1,
            dim_inputs_2=dim_hidden_2,
            num_heads=num_heads,
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # (bs, num_emb, dim_emb)
        embeddings = torch.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(embeddings)  # (bs, num_emb * dim_emb), (bs, num_emb * dim_emb)

        # get interactions from the two branches
        # (bs, dim_hidden_1), (bs, dim_hidden_1)
        latent_1, latent_2 = self.interaction_1(emb_1), self.interaction_2(emb_2)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)  # (bs, 1)
        outputs = F.sigmoid(logits)  # (bs, 1)

        return outputs
