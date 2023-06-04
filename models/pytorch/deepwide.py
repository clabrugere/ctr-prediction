import torch
from torch import nn
from torch.nn import functional as F

from models.pytorch.mlp import MLP


class DeepWide(nn.Module):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        num_hidden=3,
        dim_hidden=100,
        dropout=0.0,
    ):
        super().__init__()
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=dim_embedding)
        self.wide = nn.Embedding(num_embeddings=num_embedding, embedding_dim=1)
        
        self.interaction_mlp = MLP(
            dim_in=dim_input * dim_embedding, 
            num_hidden=num_hidden, 
            dim_hidden=dim_hidden, 
            dim_out=1, 
            dropout=dropout
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # (bs, num_emb, dim_emb)
        embeddings = torch.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))  # (bs, num_emb * dim_emb)

        latent_deep = self.interaction_mlp(embeddings)  # (bs, 1)
        latent_wide = torch.sum(self.wide(inputs), dim=1)  # (bs, 1)

        logits = latent_deep + latent_wide # (bs, 1)
        outputs = F.sigmoid(logits) # (bs, 1)

        return outputs
