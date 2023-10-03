import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import math

from tqdm import tqdm, trange


from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

LR = 0.005
NUM_EPOCHS = 40

class PatchEmbedding(nn.Module):
    def __init__(self, image_dim: tuple, patch_dim: tuple, in_channels: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        padding_x = (patch_dim[0] - (image_dim[0] % patch_dim[0])) % patch_dim[0]
        padding_y = (patch_dim[1] - (image_dim[1] % patch_dim[1])) % patch_dim[1]
        self.padding = (padding_x, padding_y)
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_dim, stride=patch_dim, padding=self.padding)

    def forward(self, x):
        n = x.shape[0]
        x = self.projection(x)
        x = x.flatten(start_dim=-2)
        x = torch.transpose(x, -2, -1)
        return x



class ViT(nn.Module):
    def __init__(self, image_dim: tuple, device, in_channels=3, n_encoders=1, patch_dim=(16, 16), hidden_dim=512, n_heads=8, out_dim=10):
        super().__init__()
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.device = device
        self.n_encoders = n_encoders
        n_patches_x = image_dim[0] // patch_dim[0] if image_dim[0] % patch_dim[0] == 0 else image_dim[0] // patch_dim[0] + 1
        n_patches_y = image_dim[1] // patch_dim[1] if image_dim[1] % patch_dim[1] == 0 else image_dim[1] // patch_dim[1] + 1
        self.n_patches = n_patches_x * n_patches_y
        self.hidden_dim = hidden_dim

        """
        # USE THIS WHEN YOU WANT THE SINE-COSINE POSITION EMBEDDINGS INSTEAD OF THE LEARNED POSITION EMBEDDINGS
        
        self.pos_embed = nn.Parameter(self.get_pos_embed(self.n_patches + 1))
        self.pos_embed.requires_grad = False

        """
        self.pos_embed = nn.Parameter(torch.rand(1, self.n_patches + 1, self.hidden_dim))
        self.class_token = nn.Parameter(torch.rand((1, self.hidden_dim)))
        self.encoders = nn.ModuleList([Encoder(self.hidden_dim, self.n_heads) for _ in range(self.n_encoders)])
        self.encoders = nn.Sequential(*(self.encoders))
        self.mlp = nn.Linear(self.hidden_dim, self.out_dim)
        self.patchify = PatchEmbedding(self.image_dim, self.patch_dim, self.in_channels, self.hidden_dim)

    def get_pos_embed(self, n_patches: int):
        result = torch.empty((n_patches, self.hidden_dim))
        for i in range(n_patches):
            for j in range(self.hidden_dim):
                result[i][j] = math.sin(i / (10000 ** (j / self.hidden_dim))) if j % 2 == 0 else math.cos(i / (10000 ** ((j - 1) / self.hidden_dim)))
        return result
    
    def forward(self, x):
        n = x.shape[0]
        patch_embeddings = self.patchify(x)
        assert patch_embeddings.shape == (n, self.n_patches, self.hidden_dim)
        embeddings = torch.empty(n, patch_embeddings.shape[1] + 1, self.hidden_dim).to(self.device)
        for i in range(n):
            embeddings[i] = torch.cat([patch_embeddings[i], self.class_token])
            embeddings[i] = embeddings[i] + self.pos_embed
        features = self.encoders(patch_embeddings)[:, 0]
        return self.mlp(features)
        
        

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=512, n_heads=8):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.v_dim = self.hidden_dim // self.n_heads
        self.Q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.Q = nn.ModuleList([nn.Linear(self.hidden_dim, self.v_dim) for _ in range(self.n_heads)])
        # self.K = nn.ModuleList([nn.Linear(self.hidden_dim, self.v_dim) for _ in range(self.n_heads)])
        # self.V = nn.ModuleList([nn.Linear(self.hidden_dim, self.v_dim) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = nn.Linear(self.hidden_dim, self.hidden_dim)

    def attention(self, query, key, value):
        scores = self.softmax((query @ key.transpose(-2, -1)) / math.sqrt(self.v_dim))
        return scores @ value

    def forward(self, x):
        query = self.Q(x).reshape(x.shape[0], -1, self.n_heads, self.v_dim).transpose(1, 2)
        key = self.K(x).reshape(x.shape[0], -1, self.n_heads, self.v_dim).transpose(1, 2)
        value = self.V(x).reshape(x.shape[0], -1, self.n_heads, self.v_dim).transpose(1, 2)
        x = self.attention(query, key, value)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.hidden_dim)
        return self.mlp(x)

    # def forward(self, sequences):
    #     results = []
    #     for sequence in sequences:
    #         seq_result = []
    #         for head in range(self.n_heads):
    #             q = self.Q[head](sequence)
    #             k = self.K[head](sequence)
    #             v = self.V[head](sequence)
    #             scores = self.softmax((q @ k.T) / math.sqrt(self.v_dim))
    #             z = scores @ v
    #             seq_result.append(z)
    #         results.append(seq_result)
    #     results = [torch.cat([head_result for head_result in seq_result], dim=-1) for seq_result in results]
    #     results = torch.cat([result[None, :] for result in results], dim=0)
    #     results = self.mlp(results)

        # return results
    


class Faster_MultiHeadAttention(nn.Module):
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
        self.mlp = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sequences):
        results = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q = self.Q[head](sequence)
                k = self.K[head](sequence)
                v = self.V[head](sequence)
                scores = self.softmax((q @ k.T) / math.sqrt(self.v_dim))
                z = scores @ v
                seq_result.append(z)
            results.append(seq_result)
        results = [torch.cat([head_result for head_result in seq_result], dim=-1) for seq_result in results]
        results = torch.cat([result[None, :] for result in results], dim=0)
        results = self.mlp(results)
        return results
    


class Encoder(nn.Module):
    def __init__(self, hidden_dim=512, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.attention = MultiHeadAttention(self.hidden_dim, self.n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
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
    


if __name__ == "__main__":
    transform = ToTensor()
    train_set = MNIST(root='./datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./datasets', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=10)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    # device = torch.device("cpu")
    model = ViT((28, 28), device, in_channels=1, n_encoders=1, patch_dim=(4, 4)).to(device)
    optimizer = opt.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        # Create a tqdm progress bar for the training batches
        with tqdm(train_loader, unit="batch") as t_bar:
            for x, y in t_bar:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=-1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                # Update tqdm progress bar description
                t_bar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
                t_bar.set_postfix(loss=total_loss / (total + 1e-8), accuracy=100 * correct / total)

        # Print epoch-level information
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss / (total + 1e-8):.4f}, Accuracy: {100 * correct / total:.2f}%")
