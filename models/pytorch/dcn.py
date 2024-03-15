import math

import torch
from torch import nn
from torch.nn import functional as F

from models.pytorch.mlp import MLP


class DCN(nn.Module):
    __CROSS_VARIANTS = ["cross", "cross_mix"]
    __AGGREGATION_MODES = ["add", "concat"]

    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding,
        num_interaction,
        num_expert,
        dim_low,
        num_hidden,
        dim_hidden,
        dropout=0.0,
        parallel_mlp=True,
        cross_type="cross_mix",
        aggregation_mode="add",
    ):
        super().__init__()

        if cross_type not in self.__CROSS_VARIANTS:
            raise ValueError(f"'cross_layer' argument must be one of {self.__CROSS_VARIANTS}")

        if aggregation_mode not in self.__AGGREGATION_MODES:
            raise ValueError(f"'aggregation_mode' must be one of {self.__AGGREGATION_MODES}")

        self.parallel_mlp = parallel_mlp
        self.aggregation_mode = aggregation_mode
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=dim_embedding)

        self.interaction_mlp = MLP(
            dim_in=dim_input * dim_embedding,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=1 if aggregation_mode == "add" else None,
            dropout=dropout,
        )

        if cross_type == "cross":
            self.cross_layers = CrossLayer(dim_input=dim_input * dim_embedding, num_layers=num_interaction)
        if cross_type == "cross_mix":
            self.cross_layers = nn.Sequential(
                *[
                    CrossMixBlock(
                        dim_input=dim_input * dim_embedding, dim_low=dim_low, num_expert=num_expert, dropout=dropout
                    )
                    for _ in range(num_interaction)
                ]
            )

        if aggregation_mode == "add":
            self.projection_head = nn.Linear(dim_input * dim_embedding, 1)
        if aggregation_mode == "concat":
            self.projection_head = nn.Linear(dim_input * dim_embedding + dim_hidden, 1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # (bs, num_emb, dim_emb)
        embeddings = torch.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        cross_out = self.cross_layers(embeddings)  # (bs, num_emb * dim_emb)

        if self.parallel_mlp:
            mlp_out = self.interaction_mlp(embeddings)  # (bs, dim_hidden) or (bs, 1)
            if self.aggregation_mode == "add":
                cross_out = self.projection_head(cross_out)  # (bs, 1)
                logits = cross_out + mlp_out  # (bs, 1)
            elif self.aggregation_mode == "concat":
                latent = torch.concat((cross_out, mlp_out), dim=-1)  # (bs, dim_input * dim_emb + dim_hidden)
                logits = self.projection_head(latent)  # (bs, 1)
        else:
            cross_out = F.relu(mlp_out)  # (bs, dim_input * dim_embedding)
            latent = self.interaction_mlp(cross_out)  # (bs, dim_hidden)
            logits = self.projection_head(latent)  # (bs, 1)

        outputs = F.sigmoid(logits)  # (bs, 1)

        return outputs


class CrossLayer(nn.Module):
    def __init__(self, dim_input, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_input, dim_input) for _ in range(num_layers)])

    def forward(self, inputs):
        outputs = inputs
        for linear in self.layers:
            outputs = inputs * linear(outputs) + outputs

        return outputs


class CrossMixLayer(nn.Module):
    def __init__(self, dim_input, dim_low):
        super().__init__()

        self.V = nn.Parameter(torch.empty((dim_input, dim_low)))
        self.C = nn.Parameter(torch.empty((dim_low, dim_low)))
        self.U = nn.Parameter(torch.empty((dim_low, dim_input)))
        self.b = nn.Parameter(torch.empty(dim_input))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.V, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.C, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.U, nonlinearity="relu")

        # bias
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.U)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, inputs):
        # x : (bs, d)
        outputs = F.relu(inputs @ self.V)  # (bs, d_low)
        outputs = F.relu(outputs @ self.C)  # (bs, d_low)
        outputs = outputs @ self.U  # (bs, d)
        outputs = inputs * outputs + self.b  # (bs, d)

        return outputs


class CrossMixBlock(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_low,
        num_expert,
        dropout=0.0,
    ):
        super().__init__()

        self.experts = nn.ModuleList([CrossMixLayer(dim_input, dim_low) for _ in range(num_expert)])
        self.gate = nn.Linear(dim_input, num_expert)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # x : (bs, d)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))

        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (bs, d, num_experts)
        gate_score = F.softmax(self.gate(inputs), dim=-1).unsqueeze(2)  # (bs, num_experts)
        outputs = self.dropout(torch.bmm(expert_outputs, gate_score).squeeze(2) + inputs)  # (bs, d)

        return outputs
