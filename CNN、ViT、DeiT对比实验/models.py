import torch
import torch.nn as nn


# ======================
# Patch Embedding
# ======================
class PatchEmbedding(nn.Module):

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=192):

        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x


# ======================
# DeiT
# ======================
class DeiT(nn.Module):

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 num_classes=10,
                 embed_dim=192,
                 depth=12,
                 num_heads=3):

        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # cls token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # distillation token
        self.dist_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 2, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # heads
        self.head = nn.Linear(embed_dim, num_classes)

        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        dist_tokens = self.dist_token.expand(B, -1, -1)

        x = torch.cat(
            (cls_tokens, dist_tokens, x),
            dim=1
        )

        x = x + self.pos_embed

        x = self.transformer(x)

        cls_out = x[:, 0]

        dist_out = x[:, 1]

        cls_logits = self.head(cls_out)

        dist_logits = self.head_dist(dist_out)

        return cls_logits, dist_logits