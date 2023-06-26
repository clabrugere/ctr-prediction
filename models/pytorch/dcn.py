import math
import torch
from torch import nn
from torch.nn import functional as F

from models.pytorch.mlp import MLP


__CROSS_VARIANTS = ["cross", "cross_mix"]
__AGGREGATION_MODES = ["add", "concat"]


class DCN(nn.Module):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_interaction=2,
        num_expert=1,
        dim_low=32,
        num_hidden=2,
        dim_hidden=16,
        dropout=0.0,
        parallel_mlp=True,
        cross_type="cross_mix",
        aggregation_mode="add",
    ):
        super().__init__()

        if cross_type not in __CROSS_VARIANTS:
            raise ValueError(f"'cross_layer' argument must be one of {__CROSS_VARIANTS}")

        if aggregation_mode not in __AGGREGATION_MODES:
            raise ValueError(f"'aggregation_mode' must be one of {__AGGREGATION_MODES}")

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
            self.cross_layers = CrossLayer(dim_in=dim_input, num_layers=num_interaction)
        if cross_type == "cross_mix":
            self.cross_layers = CrossLayerV2(
                dim_in=dim_input, dim_low=dim_low, num_expert=num_expert, num_layers=num_interaction
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
                attn_out = self.projection_head(cross_out)  # (bs, 1)
                logits = attn_out + mlp_out  # (bs, 1)
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
    def __init__(self, dim_in, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_in) for _ in range(num_layers)])

    def forward(self, inputs):
        outputs = inputs
        for linear in self.layers:
            outputs = inputs * linear(outputs) + outputs

        return outputs


class CrossLayerV2(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_low,
        num_expert=1,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            experts = nn.ModuleList()
            for _ in range(num_expert):
                V = nn.Parameter(torch.empty((dim_input, dim_low)))
                C = nn.Parameter(torch.empty((dim_low, dim_low)))
                U = nn.Parameter(torch.empty((dim_low, dim_input)))
                b = nn.Parameter(torch.empty(dim_input))

                experts.append((V, C, U, b))

            gate = nn.Parameter(torch.ones(num_expert))
            self.layers.append((experts, gate, nn.Dropout(dropout)))

        self._reset_parameters()

    def _reset_parameters(self):
        for experts, _ in self.layers:
            for V, C, U, b in experts:
                nn.init.kaiming_uniform_(V, nonlinearity="relu")
                nn.init.kaiming_uniform_(C, nonlinearity="relu")
                nn.init.kaiming_uniform_(U, nonlinearity="relu")

                # bias
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(U)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)

    def forward(self, inputs):
        # x : (bs, d)
        # projects input to a lower dimensional space then back with non-linearity in the middle
        x_0 = inputs
        x_l = inputs

        for experts, gate, dropout in self.layers:
            expert_outputs = []
            for V, C, U, b in experts:
                low_rank_proj = F.relu(x_l @ V)  # (bs, d_low)
                low_rank_inter = F.relu(low_rank_proj @ C)  # (bs, d_low)
                expert_output = x_0 * low_rank_inter @ U + b  # (bs, d)

                expert_outputs.append(expert_output)

            # aggregate expert representations and add residual connection
            gate_score = F.softmax(gate)  # (num_experts)
            expert_outputs = torch.stack(expert_outputs, dim=-1)  # (bs, d, num_experts)
            outputs = expert_outputs @ gate_score + x_l  # (bs, d)
            outputs = dropout(outputs)  # (bs, d)

            x_l = outputs

        return outputs
