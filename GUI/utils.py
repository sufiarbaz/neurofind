import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.video import r3d_18
# from DINO3D.dinov2.models.vision_transformer import vit_base_3d


def get_points_stack2(stack):
    stack_norm = cv2.normalize(stack, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = np.zeros_like(stack_norm, dtype=np.uint8)
    for i in range(stack_norm.shape[0]):
        mask[i] = cv2.threshold(stack_norm[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    points = np.argwhere(mask > 0)
    return points.tolist()

def crop_around_point(stack, point, size_x=64, size_y=64, size_z=9):
    """
    Crop a single 3D ROI from 'image' (Z, Y, X).
    point: (z, y, x) center point (ints or castable to int).
    size_x, size_y, size_z: desired output sizes along X, Y, Z.
    Returns an array of shape (size_z, size_y, size_x) (padded with the minimum
    value of the input image if near borders).
    """
    z_dim, y_dim, x_dim = stack.shape
    zc, yc, xc = map(int, point)

    # compute start/end for each axis such that (end - start) == size
    sx = xc - (size_x // 2)
    ex = sx + size_x
    sy = yc - (size_y // 2)
    ey = sy + size_y
    sz = zc - (size_z // 2)
    ez = sz + size_z

    # clamp to image bounds
    sx_clamped = max(sx, 0)
    ex_clamped = min(ex, x_dim)
    sy_clamped = max(sy, 0)
    ey_clamped = min(ey, y_dim)
    sz_clamped = max(sz, 0)
    ez_clamped = min(ez, z_dim)

    # slice
    cropped = stack[sz_clamped:ez_clamped, sy_clamped:ey_clamped, sx_clamped:ex_clamped]

    # if already correct size, return
    if cropped.shape == (size_z, size_y, size_x):
        return torch.from_numpy(cropped[np.newaxis, np.newaxis, :]) # add batch and channel dims

    # pad with reflection padding
    pad_z = (max(0, -sz), max(0, ez - z_dim))
    pad_y = (max(0, -sy), max(0, ey - y_dim))
    pad_x = (max(0, -sx), max(0, ex - x_dim))
    
    padded = np.pad(cropped, 
                    (pad_z, pad_y, pad_x), 
                    mode='reflect')
    
    # print(padded.shape, (size_z, size_y, size_x))

    # plt.imshow(padded[padded.shape[0]//2], cmap='gray')
    # plt.title(f"Cropped and padded volume (Z={size_z}, Y={size_y}, X={size_x})")    
    # plt.show()

    return torch.from_numpy(padded[np.newaxis, np.newaxis, :])  # add batch and channel dims

def compute_embedding(crop, model, device):
    model.eval()
    with torch.no_grad():
        crop = crop.to(device)
        emb = model(crop)        # [1, 128]
        emb = emb.squeeze(0)     # [128]
        emb = emb.cpu()
    return emb


def compare_embeddings(emb1, emb2_list):
    """
    emb1: Tensor [1, D] or [N1, D]
    emb2_list: List[Tensor[D]] or List[np.ndarray]
    """

    # print(type(emb2_list), len(emb2_list))
    # print(type(emb2_list[0]), emb2_list[0].shape)

    if isinstance(emb2_list, np.ndarray):
        emb2 = torch.stack([
            e if torch.is_tensor(e) else torch.from_numpy(e)
            for e in emb2_list
        ], dim=0)
    else:
        emb2 = emb2_list

    if emb1.ndim == 1:
        emb1 = emb1.unsqueeze(0)

    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    sim = emb1 @ emb2.T  # [1, N2]
    best_idx = sim.argmax(dim=1)

    return best_idx.item(), sim.squeeze(0)

def to_uint8(img):
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)

class DINOv3Encoder(nn.Module):
    def __init__(self, embedding_dim=128, model='dinov3_vitb16', freeze_backbone=False):
        super().__init__()

        self.backbone = torch.hub.load(
            'dinov3', 
            model,
            source='local', 
            pretrained=False)
        
        old_proj = self.backbone.patch_embed.proj

        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )

        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        self.backbone.patch_embed.proj = new_proj

        feat_dim = self.backbone.embed_dim  # typically 768 for vit-b

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fc = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Frame-wise encoding
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        feats = self.backbone(x)            # [B*T, feat_dim]
        feats = self.fc(feats)              # [B*T, D]

        feats = feats.view(B, T, -1).mean(dim=1)  # temporal pooling

        return F.normalize(feats, dim=-1)   # [B, D]
    
class ResNet3DEncoder(nn.Module):
    def __init__(self, embedding_dim=128, weights=None, use_multichannel=True):
        super().__init__()
        self.use_multichannel = use_multichannel
        self.backbone = r3d_18(weights=weights)

        # Adapt first conv layer if using only 1 channel
        if not use_multichannel:
            old_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

        self.backbone.fc = nn.Identity()  # remove classifier
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        feats = self.backbone(x)          # [B, 512]
        return F.normalize(self.fc(feats), dim=-1)  # [B, D]
    
# class DINO3DEncoder(nn.Module):
#     def __init__(self, embedding_dim=128, freeze_backbone=False):
#         super().__init__()

#         # 3DINO ViT backbone
#         self.backbone = vit_base_3d(
#             patch_size=16,
#             in_chans=1,
#         )

#         feat_dim = self.backbone.embed_dim  # usually 768

#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False

#         self.fc = nn.Linear(feat_dim, embedding_dim)

#     def forward(self, x):
#         """
#         x: [B, 1, T, H, W]
#         """

#         # ---- PAD so patch sizes divide volume ----
#         _, _, D, H, W = x.shape
#         pad_d = (16 - D % 16) % 16
#         pad_h = (16 - H % 16) % 16
#         pad_w = (16 - W % 16) % 16

#         if pad_d or pad_h or pad_w:
#             x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

#         # ---- forward through 3DINO ----
#         out = self.backbone.forward_features(x)

#         # CLS token (recommended)
#         feats = out["x_norm_clstoken"]   # [B, C]

#         # Alternative (often better):
#         # feats = out["x_norm_patchtokens"].mean(dim=1)

#         feats = self.fc(feats)
#         return F.normalize(feats, dim=-1)