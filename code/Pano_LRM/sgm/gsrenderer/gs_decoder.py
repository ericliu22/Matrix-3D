
import math
import copy
import torch
import torch.nn as nn
from sat.model.mixins import BaseMixin
# from sat.sgm.transformer import Transformer
from .dpt_head import PixelwiseTaskWithDPT

class TokenDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.layernorm = nn.LayerNorm(1024, bias=False)

        self.deconv3d = nn.ConvTranspose3d(
            in_channels=1024, out_channels=14,
            kernel_size=(5,8,8), stride=(4,8,8), padding=(2,0,0), bias=False
        )

        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(), nn.Linear(1024, 2 * 1024, bias=True)
        # )

    def forward(self, img_tokens, T, h, w):
        """
        img_tokens: [b, n_patches, d]
        t_embedding: [b, d]
        output: [b, n_patches, dd]
        """
        # shift, scale = self.adaLN_modulation(t_embedding).chunk(2, dim=1)
        # img_tokens = modulate(self.layernorm(img_tokens), shift, scale)
        img_tokens = img_tokens.reshape(-1, T, h//2, w//2, 1024)
        img_tokens = img_tokens.permute(0, 4, 1, 2, 3)
        img_tokens = self.deconv3d(img_tokens)
        return img_tokens

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _init_weights_t(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight)
        nn.init.zeros_(module.out_proj.bias)



class GS_decoder(nn.Module):
    def __init__(self, gs_dim=14):
        super().__init__()

        self.video_conv = nn.Conv2d(in_channels=16, out_channels=1024, kernel_size=2, stride=2)
        #self.video_conv = nn.Conv2d(in_channels=32, out_channels=1024, kernel_size=4, stride=4, padding=(0,2))
        self.video_layer_norm = nn.LayerNorm(1024)
        self.video_conv.apply(_init_weights)

        # Camera embedding Conv3D
        self.camera_conv = nn.Conv3d(in_channels=6, out_channels=1024, kernel_size=(4,16,16), stride=(4,16,16), padding=(2,0,0))
        #self.camera_conv = nn.Conv3d(in_channels=6, out_channels=1024, kernel_size=(4,32,32), stride=(4,32,32), padding=(2,0,16))
        self.camera_layer_norm = nn.LayerNorm(1024)
        self.camera_conv.apply(_init_weights)

        # Linear projection
        self.linear_proj = nn.Linear(1024*2, 1024)
        self.linear_proj.apply(_init_weights)

        self.transformer_input_layernorm = nn.LayerNorm(1024, bias=False)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=4,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # important: makes input [B, N, C]
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.transformer.apply(_init_weights)

        self.transformer_gs = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.transformer_gs.apply(_init_weights)

        # self.deconv3d = nn.ConvTranspose3d(
        #     in_channels=768, out_channels=gs_dim,
        #     kernel_size=(5,8,8), stride=(4,8,8), padding=(2,0,0), bias=False
        # )
        # self.deconv3d.apply(_init_weights)

        self.deconv3d_dpt = nn.ConvTranspose3d(
            in_channels=256, out_channels=3,
            kernel_size=(5,1,1), stride=(4,1,1), padding=(2,0,0), bias=False
        )
        self.deconv3d_dpt.apply(_init_weights)

        # self.deconv3d_depth = nn.ConvTranspose3d(
        #     in_channels=768, out_channels=3,
        #     kernel_size=(5,8,8), stride=(4,8,8), padding=(2,0,0), bias=False
        # )
        # self.deconv3d_depth.apply(_init_weights)

        self.deconv3d_sh = nn.ConvTranspose3d(
            in_channels=256, out_channels=gs_dim,
            kernel_size=(5,1,1), stride=(4,1,1), padding=(2,0,0), bias=False
        )
        self.deconv3d_sh.apply(_init_weights)

        self.dpt_head = PixelwiseTaskWithDPT(num_channels=1, feature_dim=1024, last_dim=512, hooks_idx=[0, 1, 2, 3],dim_tokens=[1024,1024,1024,1024],head_type='regression')
        self.dpt_head.apply(_init_weights)

        self.dpt_head_gs = PixelwiseTaskWithDPT(num_channels=gs_dim, feature_dim=1024, last_dim=512, hooks_idx=[0, 1, 2, 3],dim_tokens=[1024,1024,1024,1024],head_type='regression')
        self.dpt_head_gs.apply(_init_weights)

    def forward(self, latents, plucker):
        b, T, c, h, w = latents.size()
        latent_tokens = self.video_conv(latents.flatten(0,1))
        latent_tokens = latent_tokens.flatten(start_dim=2)
        latent_tokens = latent_tokens.transpose(1, 2)
        latent_tokens = self.video_layer_norm(latent_tokens)
        latent_tokens = latent_tokens.reshape(b, -1, 1024)

        # Camera convolution
        camera_out = self.camera_conv(plucker.permute(0,4,1,2,3))  # [B, 1024, N_l]
        
        camera_out = camera_out.flatten(start_dim=2)
        camera_out = camera_out.permute(0, 2, 1)

        # Concatenate features

        combined_tokens = torch.cat((latent_tokens, camera_out), dim=-1)  # [B, N, 2048]
        # Linear projection
        combined_tokens = self.linear_proj(combined_tokens)  # [B, N, 1024]
        combined_tokens = self.transformer_input_layernorm(combined_tokens)

        # transformer_out = self.transformer(combined_tokens) 
        # transformer_out_gs = self.transformer_gs(transformer_out)

        all_layers = []
        all_layers_gs = []

        for layer in self.transformer.layers:  # 每层是 TransformerEncoderLayer
            combined_tokens = layer(combined_tokens)
            all_layers.append(combined_tokens)

        for layer in self.transformer_gs.layers:  # 每层是 TransformerEncoderLayer
            combined_tokens = layer(combined_tokens)
            all_layers_gs.append(combined_tokens)
            
        res1 = self.dpt_head( all_layers, [60, 120])
        res1 = res1.permute(1,0,2,3)[None]

        gs_para = self.dpt_head_gs(all_layers_gs, [60, 120])
        gs_para = gs_para.permute(1,0,2,3)[None]

      
        # transformer_out = transformer_out.reshape(-1, T, h//2, w//2, 1024)
        # transformer_out = transformer_out.permute(0, 4, 1, 2, 3).contiguous()
    
        # transformer_out_gs = transformer_out_gs.reshape(-1, T, h//2, w//2, 1024)
        # transformer_out_gs = transformer_out_gs.permute(0, 4, 1, 2, 3).contiguous()
        # gs = self.deconv3d(transformer_out_gs)
        # gs_para = gs.permute(0,2,3,4,1)

        # gs_depth = self.deconv3d_depth(transformer_out)
        # gs_depth = gs_depth.permute(0,2,3,4,1)

        gs_depth = self.deconv3d_dpt(res1)
        gs_depth = gs_depth.permute(0,2,3,4,1)

        gs_para = self.deconv3d_sh(gs_para)
        gs_para = gs_para.permute(0,2,3,4,1)

        return gs_para, gs_depth

class PixelShuffleTimeSpace(nn.Module):
    def __init__(self, upscale_t=2, upscale_h=2, upscale_w=2):
        super().__init__()
        self.rt = upscale_t
        self.rh = upscale_h
        self.rw = upscale_w

    def forward(self, x):
        B, C, T, H, W = x.shape
        rt, rh, rw = self.rt, self.rh, self.rw
        r3 = rt * rh * rw
        assert C % r3 == 0, f"C ({C}) not divisible by upscale factors ({r3})"

        C_out = C // r3
        x = x.view(B, C_out, rt, rh, rw, T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(B, C_out, T * rt, H * rh, W * rw)
        return x


class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.scale = (channels // 8) ** -0.5

    def forward(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        x_ = x.view(B, T * H * W, C)  # [B, N, C]
        x_ = self.norm(x_)
        qkv = self.qkv(x_).chunk(3, dim=-1)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = self.proj(out)
        out = out.view(B, T, H, W, C)
        return out + x  # residual


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factors):
        super().__init__()
        self.r_t, self.r_h, self.r_w = upscale_factors

    def forward(self, x):
        B, C_mul, T, H, W = x.shape
        r_t, r_h, r_w = self.r_t, self.r_h, self.r_w
        C = C_mul // (r_t * r_h * r_w)

        x = x.view(B, C, r_t, r_h, r_w, T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)  # [B, C, T, r_t, H, r_h, W, r_w]
        x = x.reshape(B, C, T * r_t, H * r_h, W * r_w)
        return x

class Upsampler3DWithAttention(nn.Module):
    def __init__(self, in_channels=1024, out_channels=14, upscale=(4, 8, 8)):
        super().__init__()
        r_t, r_h, r_w = upscale
        self.conv = nn.Conv3d(
            in_channels, out_channels * r_t * r_h * r_w,
            kernel_size=1, stride=1, padding=0
        )
        self.shuffle = PixelShuffle3D(upscale)
        #self.attn = SpatioTemporalAttention(out_channels)

    def forward(self, x):
        # [B, T, H, W, C] → [B, C, T, H, W]
        x = x.permute(0, 4, 1, 2, 3)

        x = self.conv(x)
        x = self.shuffle(x)
        # [B, C, T, H, W] → [B, T, H, W, C]
        x = x.permute(0, 2, 3, 4, 1)
        #x = self.attn(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
class ImageTokenDecoder(BaseMixin):
    def __init__(self, dim, decode_dim):
        super().__init__()
        self.dim = dim
        self.decode_dim = decode_dim
        self.dec_emb = nn.Linear(self.dim, self.decode_dim)
        self.layernorm = nn.LayerNorm(self.decode_dim, bias=False)

        self.upsamper = Upsampler3DWithAttention(in_channels=self.decode_dim, out_channels=14, upscale=(4, 8, 8))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(512, 2 * self.decode_dim, bias=True)
        )

    def gs_decoder(self, img_tokens, t_embedding, T, h, w, **kwargs):
        """
        img_tokens: [b,  d] [17550, 3072]
        t_embedding: [b, d]
        output: [b, n_patches, dd]
        """
        shift, scale = self.adaLN_modulation(t_embedding).chunk(2, dim=1)
        img_tokens = modulate(self.layernorm(img_tokens), shift, scale)
        img_tokens = img_tokens.reshape(-1, T, h//2, w//2, self.dim)
        img_tokens = img_tokens.permute(0, 4, 1, 2, 3)
        img_tokens = self.deconv3d(img_tokens)
        return img_tokens

    def forward(self, img_tokens, t_embedding, T, h, w):
        """
        img_tokens: [b, n_patches, d]
        t_embedding: [b, d]
        output: [b, n_patches, dd]
        """
        
        dec_emb = self.dec_emb(img_tokens)
        shift, scale = self.adaLN_modulation(t_embedding).chunk(2, dim=1)
        dec_emb = modulate(self.layernorm(dec_emb), shift, scale)
        # reshape to spatial representation
        dec_emb = dec_emb.reshape(-1, T, h//2, w//2, self.decode_dim)
        dec_emb = self.upsamper(dec_emb)
        dec_emb = dec_emb[:,:49]
        # upsampler using pixel shuffle

        #img_tokens = img_tokens.permute(0, 4, 1, 2, 3)
    
        return dec_emb

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.upsamper.conv.weight)


class GaussiansUpsampler(nn.Module):
    def __init__(self, gaussians_sh_degree):
        super().__init__()
        self.gaussians_sh_degree = gaussians_sh_degree
        """
        xyz : torch.tensor of shape (n_gaussians, 3)
        features : torch.tensor of shape (n_gaussians, (sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (n_gaussians, 3)
        rotation : torch.tensor of shape (n_gaussians, 4)
        opacity : torch.tensor of shape (n_gaussians, 1)
        """

    def forward(self, gaussians):     # 把 dimenssion 分掉，分出对应的维度
        """
        gaussians: [b, n_gaussians, d]
        n_gaussians - 高斯的数量
        d - 每个高斯的 attribute 的数量
        """
        img_aligned_gaussians = gaussians.reshape(1, gaussians.shape[1], -1) #[b, 14, -1]
        img_aligned_gaussians = img_aligned_gaussians.transpose(1,2)#

        xyz, features, scaling, rotation, opacity = img_aligned_gaussians.split( [3, (self.gaussians_sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=2)
        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.gaussians_sh_degree + 1) ** 2,
            3,
        )
        # torch.exp(torch.tensor(-2.3)) = 0.1
        # torch.exp(torch.tensor(-1.2)) = 0.3
        scaling = (scaling - 2.3).clamp(max=-1.20)
        # torch.sigmoid(-torch.tensor(2.)) = 0.1192
        opacity = opacity - 2.0
        return xyz, features, scaling, rotation, opacity

