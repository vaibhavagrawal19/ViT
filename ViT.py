import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import math

from tqdm import tqdm, trange

class PatchEmbedding(nn.Module):
    def __init__(self, image_dim: tuple, patch_dim: tuple, in_channels: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding = (patch_dim[0] - (image_dim[0] % patch_dim[0]), patch_dim[1] - (image_dim[1] % patch_dim[1]))
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_dim, stride=patch_dim, padding=self.padding)

    def forward(self, x):
        n = x.shape[0]
        x = self.projection(x)
        x = x.reshape((n, -1, self.embed_dim))
        return x
        

class ViT(nn.Module):
    def __init__(self, image_dim: tuple, in_channels=3, patch_dim=(16, 16), hidden_dim=512):
        super().__init__()
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.class_token = nn.Parameter(torch.rand(1, self.embed_dim))
        n_patches_x = image_dim[0] // patch_dim[0] if image_dim[0] % patch_dim[0] == 0 else image_dim[0] // patch_dim[0] + 1
        n_patches_y = image_dim[1] // patch_dim[1] if image_dim[1] % patch_dim[1] == 0 else image_dim[1] // patch_dim[1] + 1
        self.n_patches = n_patches_x * n_patches_y
        self.hidden_dim = hidden_dim
        self.pos_embed = nn.Parameter(self.get_pos_embed(self.n_patches + 1))
        self.pos_embed.requires_grad = False
        self.class_token = nn.Parameter(torch.empty((self.hidden_dim, )))

    def get_pos_embed(self, n_patches: int, type="manual"):
        result = torch.empty((n_patches, self.hidden_dim))
        for i in range(n_patches):
            for j in range(self.D):
                result[i][j] = torch.sin(i / (10000 ** (j / self.D))) if j % 2 == 0 else torch.cos(i / (10000 ** ((j - 1) / self.D)))
        return result
    
    def forward(self, X):
        n = X.shape[0]
        patchify = PatchEmbedding(self.image_dim, self.patch_dim, self.in_channels, self.hidden_dim)
        patch_embeddings = patchify(X)
        for i in range(n):
            patch_embeddings[i] = torch.cat([patch_embeddings[i], nn.Parameter(torch.empty((self.hidden_dim, )))])
            patch_embeddings[i] = patch_embeddings[i] + self.pos_embed
        

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=512, n_heads=8):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.v_dim = self.hidden_dim // self.n_heads
        self.Q = nn.ModuleList([nn.Linear(self.hidden_dim, self.v_dim) for _ in range(self.n_heads)])
        self.K = nn.ModuleList([nn.Linear(self.hidden_dim, self.v_dim) for _ in range(self.n_heads)])
        self.V = nn.ModuleList([nn.Linear(self.hidden_dim, self.v_dim) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        results = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q = self.Q[head](sequence)
                k = self.K[head](sequence)
                v = self.V[head](sequence)
                scores = self.softmax((q @ k.T) / torch.sqrt(self.v_dim))
                seq_result.append(v.T @ scores)
            results.append(seq_result)
        return torch.cat(results, dim=1)
    

class Encoder(nn.Module):
    def __init__(self, hidden_dim=512, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.attention = MultiHeadAttention(self.hidden_dim, self.n_heads)
        self.mlp = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        n = x.shape[0]
        x_ = x.clone()
        x = self.layer_norm(x)
        x = self.attention(x)
        x = x + x_
        x_ = x.clone()
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = x + x_
        del x_
        return x
