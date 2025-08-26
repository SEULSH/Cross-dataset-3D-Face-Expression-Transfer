
import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
from functools import partial
from xcit import XCABlock1D

class ConvPatchEmbed(nn.Module):
    def __init__(self, kernel_size=3, img_size=100, patch_size=13, in_chans=3, embed_dim=512):
        super().__init__()

        self.img_size = img_size

        self.num_patches = patch_size * patch_size

        self.proj = torch.nn.Sequential(

            nn.Conv2d(
                in_chans, embed_dim // 8, kernel_size=kernel_size, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(embed_dim // 8),
            nn.GELU(),

            nn.Conv2d(
                embed_dim // 8, embed_dim // 4, kernel_size=kernel_size, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),

            nn.Conv2d(
                embed_dim // 4, embed_dim, kernel_size=kernel_size, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)  # bs, 512, 13, 13
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # bs, 13*13, 512

        return x, (Hp, Wp)

class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        # self.bn = nn.SyncBatchNorm(in_features)
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x

class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #
        # qc = q[:, :, 0:1]   # CLS token
        # attn_cls = (qc * k).sum(dim=-1) * self.scale
        # attn_cls = attn_cls.softmax(dim=-1)
        # attn_cls = self.attn_drop(attn_cls)
        #
        # cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        # cls_tkn = self.proj(cls_tkn)
        # x = torch.cat([self.proj_drop(cls_tkn), x[:, 1:]], dim=1)
        # return x   # # x[:, 1:] are not changed here

        ##### changed by lsh   ## if only use this, get same neutral expression face
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qk = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_cls = qk.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        cls_tkn = torch.matmul(attn_cls, v)  # B, h, N, D
        cls_tkn = self.proj(cls_tkn.permute(0,2,1,3).reshape(B,N,-1))
        x = self.proj_drop(cls_tkn)
        return x

class ClassAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=None,
                 tokens_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.LayerNorm.DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if eta is not None:     # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # FIXME: A hack for models pre-trained with layernorm over all the tokens not just the CLS
        self.tokens_norm = tokens_norm

    def forward(self, x, H, W, mask=None):
        # x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        # if self.tokens_norm:
        #     x = self.norm2(x)
        # else:
        #     x[:, 0:1] = self.norm2(x[:, 0:1].clone())  ## add clone by lsh
        #
        # x_res = x
        # cls_token = x[:, 0:1]
        # cls_token = self.gamma2 * self.mlp(cls_token)
        # x = torch.cat([cls_token, x[:, 1:]], dim=1)
        # x = x_res + self.drop_path(x)
        # return x   ## x[:, 1:] just ++++++ ???

        #### changed by lsh   ## if only use this, get same neutral expression face
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.tokens_norm:
            x = self.norm2(x)
        cls_token = self.gamma2 * self.mlp(x)
        x = x + self.drop_path(cls_token)
        return x

class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class Mlp(nn.Module):
    """
    The Mlp is from form ViT.
    we change paras settings.
    """
    def __init__(self, in_features=512, hidden_features=1024, act_layer=None, drop=0.1):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class XCABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 num_tokens=196, eta=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = nn.LayerNorm.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)  # 512 * 2 = 1024
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class XCiT(nn.Module):

    def __init__(self, kernel_size=3, img_size=100, patch_size=13, embed_dim=512,
                 depth=3, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 cls_attn_layers=3, eta=1.0, tokens_norm=False):
        super().__init__()

        self.num_features = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = ConvPatchEmbed(kernel_size=kernel_size,
                                          img_size=img_size,
                                          patch_size=patch_size,
                                          embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [drop_path_rate for i in range(depth)]

        ##
        self.blocks = nn.ModuleList([
            XCABlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, num_tokens=num_patches, eta=eta)
            for i in range(depth)])

        # we do use class attention blocks
        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                eta=eta, tokens_norm=tokens_norm)
            for i in range(cls_attn_layers)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, self.num_features)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)  # bs, 13*13, 512, (13, 13)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # bs, 1, 512
        x = torch.cat((cls_tokens, x), dim=1)  # bs, 13*13+1, 512

        for blk in self.cls_attn_blocks:  # we do use class attention blocks. if only this, get same neutral expression
            x = blk(x, Hp, Wp)

        x = self.norm(x)[:, 0]

        return x  # bs, 512

    def forward(self, x):  # bs, c, h, w
        x = self.forward_features(x)  # bs, 512
        x = self.head(x)
        return x  # bs, 512

class Im_ex_encoder(nn.Module):
    def __init__(self, kernel_size=3, patch_size=13, embed_dim=256):
        super(Im_ex_encoder, self).__init__()

        self.Encoder = XCiT(kernel_size=kernel_size, patch_size=patch_size, embed_dim=embed_dim)

    def forward(self, img):  # b, c, h, w

        encoded = self.Encoder(img)

        return encoded  # b, 512


#############################################
class Decoder(nn.Module):
    def __init__(self, bottleneck_size=256):
        self.bottleneck_size = bottleneck_size
        super(Decoder, self).__init__()

        eta = 1e-5

        self.geo_conv0 = nn.ModuleList([XCABlock1D(self.bottleneck_size, 16, eta=eta), XCABlock1D(self.bottleneck_size, 16, eta=eta)])
        self.conv0 = torch.nn.Conv1d(self.bottleneck_size + self.bottleneck_size, self.bottleneck_size, 1)
        self.conv_d0 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)

        self.geo_conv1 = nn.ModuleList([XCABlock1D(self.bottleneck_size // 2, 8, eta=eta), XCABlock1D(self.bottleneck_size // 2, 8, eta=eta)])
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size // 2 + self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv_d1 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)

        self.geo_conv2 = nn.ModuleList([XCABlock1D(self.bottleneck_size // 4, 4, eta=eta), XCABlock1D(self.bottleneck_size // 4, 4, eta=eta)])
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size // 4 + self.bottleneck_size, self.bottleneck_size // 4, 1)
        self.conv_d2 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()

    def forward(self, x2, ex_f):   # bs, D, N; bs, 512

        N = x2.shape[2]
        ex_f = ex_f.unsqueeze(1).repeat(1, N, 1)

        x2_f = x2.transpose(1, 2)
        for m in self.geo_conv0:
            x2_f = m(x2_f)
        dx = F.elu(self.conv0(torch.cat([x2_f, ex_f], dim=-1).transpose(1, 2)))
        x2 = self.conv_d0(x2 + dx)

        x2_f = x2.transpose(1, 2)
        for m in self.geo_conv1:
            x2_f = m(x2_f)
        dx = F.elu(self.conv1(torch.cat([x2_f, ex_f], dim=-1).transpose(1, 2)))
        x2 = self.conv_d1(x2 + dx)

        x2_f = x2.transpose(1, 2)
        for m in self.geo_conv2:
            x2_f = m(x2_f)
        dx = F.elu(self.conv2(torch.cat([x2_f, ex_f], dim=-1).transpose(1, 2)))
        x2 = self.conv_d2(x2 + dx)

        # x2 = 2 * self.th(x2)
        x2 = self.th(x2)

        return x2

class Identity_encoder(nn.Module):
    def __init__(self):
        super(Identity_encoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.th = nn.Tanh()

    def forward(self, x):
        x = self.th(self.conv1(x))
        x = self.th(self.conv2(x))
        x = self.th(self.conv3(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, bottleneck_size=256):
        super(Autoencoder, self).__init__()

        self.bottleneck_size = bottleneck_size

        self.identity_encoder = Identity_encoder()

        self.encoder_level0 = Im_ex_encoder(kernel_size=3, patch_size=13)
        self.encoder_level1 = Im_ex_encoder(kernel_size=5, patch_size=11)
        self.encoder_level2 = Im_ex_encoder(kernel_size=7, patch_size=9)

        self.decoder = Decoder(bottleneck_size=self.bottleneck_size)


    def forward(self, x1, x2):  # s: bs,c,h,w, t: bs, N, 3

        x2 = x2.transpose(1, 2)
        x2 = self.identity_encoder(x2)  # bs, D, N

        x11 = self.encoder_level0(x1)  # bs, 256
        x12 = self.encoder_level1(x1)  # bs, 256
        x13 = self.encoder_level2(x1)  # bs, 256

        out = 1.0/3 * self.decoder(x2, x13) + 1.0/3 * self.decoder(x2, x12) + 1.0/3 * self.decoder(x2, x11)

        return out.transpose(2, 1)