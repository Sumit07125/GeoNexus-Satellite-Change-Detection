
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

CH_RAW = slice(0, 11)
CH_DERIVED = slice(11, 13)
CH_SAR = slice(13, 16)
CH_Q = slice(16, 17)

SSL4EO_S2_ORDER = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
OUR_S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']

class SplitStem(nn.Module):
    def __init__(self, n_raw=11, n_derived=2, out_ch=64, use_derived=True):
        super().__init__()
        self.use_derived = use_derived
        self.raw = nn.Conv2d(n_raw, out_ch, 7, stride=2, padding=3, bias=False)
        self.derived = nn.Conv2d(n_derived, out_ch, 7, stride=2, padding=3, bias=False)
        nn.init.zeros_(self.derived.weight)
    def forward(self, x_raw, x_derived):
        y = self.raw(x_raw)
        if self.use_derived:
            y = y + self.derived(x_derived)
        return y

class BranchEncoder(nn.Module):
    def __init__(self, n_raw, n_derived, use_derived=True):
        super().__init__()
        r = resnet18(weights=None)
        self.stem = SplitStem(n_raw, n_derived, 64, use_derived)
        self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.layer3, self.layer4 = r.layer3, r.layer4
    def forward(self, x_raw, x_derived):
        x = self.maxpool(self.relu(self.bn1(self.stem(x_raw, x_derived))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

class DualEncoder(nn.Module):
    def __init__(self, use_derived=True):
        super().__init__()
        self.optical = BranchEncoder(11, 2, use_derived)
        self.sar = BranchEncoder(3, 1, use_derived=False)
    def forward(self, x):
        f_opt = self.optical(x[:, CH_RAW], x[:, CH_DERIVED])
        zeros = torch.zeros(x.shape[0], 1, *x.shape[2:], device=x.device, dtype=x.dtype)
        f_sar = self.sar(x[:, CH_SAR], zeros)
        return f_opt, f_sar

class QualityGate(nn.Module):
    def __init__(self, channels=(64,128,256,512)):
        super().__init__()
        self.gates = nn.ModuleList([nn.Conv2d(2*c+1, 1, 1) for c in channels])
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, 2.0)
        self.mode = 'gated'
    def forward(self, f_opt, f_sar, q):
        fused, gates = [], []
        for i, (fo, fs) in enumerate(zip(f_opt, f_sar)):
            if self.mode == 'optical_only':
                fused.append(fo)
                gates.append(torch.ones_like(fo[:, :1]))
                continue
            qs = F.adaptive_avg_pool2d(q, fo.shape[-2:])
            g = torch.sigmoid(self.gates[i](torch.cat([fo, fs, qs], dim=1)))
            fused.append(g * fo + (1.0-g) * fs)
            gates.append(g)
        return fused, gates

def gate_loss(gates, q):
    loss = 0.0
    for g in gates:
        qs = F.adaptive_avg_pool2d(q, g.shape[-2:])
        loss = loss + F.mse_loss(g, qs)
    return loss / len(gates)

class TemporalDiff(nn.Module):
    def forward(self, f1_list, f2_list):
        out = []
        for f1, f2 in zip(f1_list, f2_list):
            d_abs = (f1-f2).abs()
            d_mul = f1*f2
            d_cos = F.cosine_similarity(f1, f2, dim=1).unsqueeze(1)
            out.append(torch.cat([d_abs, d_mul, d_cos], dim=1))
        return out

class LKA(nn.Module):
    def __init__(self, dim, k=7, d=2):
        super().__init__()
        dw_k = 2*d-1
        dwd_k = (k+d-1)//d
        dwd_k = dwd_k if dwd_k % 2 == 1 else dwd_k-1
        self.dw = nn.Conv2d(dim, dim, dw_k, padding=dw_k//2, groups=dim)
        self.dwd = nn.Conv2d(dim, dim, dwd_k, padding=(dwd_k//2)*d, groups=dim, dilation=d)
        self.pw = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        return x * self.pw(self.dwd(self.dw(x)))

class LKABlock(nn.Module):
    def __init__(self, dim, k=7, d=2):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.p1, self.act = nn.Conv2d(dim, dim, 1), nn.GELU()
        self.lka, self.p2 = LKA(dim, k, d), nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        return x + self.p2(self.lka(self.act(self.p1(self.norm(x)))))

def conv_bn_gelu(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k//2, bias=False), nn.BatchNorm2d(o), nn.GELU())

class LKADecoder(nn.Module):
    def __init__(self, in_ch=(129,257,513,1025), dec_ch=(256,128,64), k=7):
        super().__init__()
        c1, c2, c3 = dec_ch
        self.bottleneck = nn.Sequential(conv_bn_gelu(in_ch[3], c1, 1), LKABlock(c1, k), LKABlock(c1, k))
        self.skip3 = conv_bn_gelu(in_ch[2], c1, 1)
        self.skip2 = conv_bn_gelu(in_ch[1], c2, 1)
        self.skip1 = conv_bn_gelu(in_ch[0], c3, 1)
        self.up3 = nn.Sequential(conv_bn_gelu(c1+c1, c1), LKABlock(c1, k))
        self.up2 = nn.Sequential(conv_bn_gelu(c1+c2, c2), LKABlock(c2, k))
        self.up1 = nn.Sequential(conv_bn_gelu(c2+c3, c3), LKABlock(c3, k))
        self.head3 = nn.Conv2d(c1, 1, 1)
        self.head2 = nn.Conv2d(c2, 1, 1)
        self.head1 = nn.Conv2d(c3, 1, 1)
    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    def forward(self, d):
        x = self.bottleneck(d[3])
        s3 = self.skip3(d[2])
        x = self.up3(torch.cat([self._up(x, s3), s3], 1))
        o3 = self.head3(x)
        s2 = self.skip2(d[1])
        x = self.up2(torch.cat([self._up(x, s2), s2], 1))
        o2 = self.head2(x)
        s1 = self.skip1(d[0])
        x = self.up1(torch.cat([self._up(x, s1), s1], 1))
        o1 = self.head1(x)
        return o1, o2, o3

class GeoNexusCD(nn.Module):
    def __init__(self, use_derived=True, decoder='lka', dec_ch=(256,128,64), k=7):
        super().__init__()
        self.encoder = DualEncoder(use_derived)
        self.gate = QualityGate((64,128,256,512))
        self.tdiff = TemporalDiff()
        in_ch = tuple(2*c+1 for c in (64,128,256,512))
        if decoder != 'lka':
            raise ValueError(decoder)
        self.decoder = LKADecoder(in_ch, dec_ch, k)
    def set_mode(self, mode):
        if mode not in ('gated','optical_only'):
            raise ValueError(mode)
        self.gate.mode = mode
        return self
    def encode(self, x):
        f_opt, f_sar = self.encoder(x)
        return self.gate(f_opt, f_sar, x[:, CH_Q])
    def forward(self, x1, x2, out_size=None):
        f1, g1 = self.encode(x1)
        f2, g2 = self.encode(x2)
        d = self.tdiff(f1, f2)
        o1, o2, o3 = self.decoder(d)
        size = out_size or x1.shape[-2:]
        up = lambda o: F.interpolate(o, size=size, mode='bilinear', align_corners=False)
        return {'logits': up(o1), 'aux8': up(o2), 'aux16': up(o3), 'gates': g1+g2}

def dice_loss(logits, target, smooth=1.0):
    p = torch.sigmoid(logits)
    num = 2*(p*target).sum(dim=(1,2,3)) + smooth
    den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + smooth
    return (1 - num/den).mean()

def bce_dice(logits, target, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return 0.5*bce + 0.5*dice_loss(logits, target)

def total_loss(out, target, q=None, w_deep=(1.0,0.5,0.25), w_gate=0.05, pos_weight=None):
    weights = torch.tensor(w_deep, device=target.device)
    weights = (weights / weights.sum()).tolist()
    l = (weights[0]*bce_dice(out['logits'], target, pos_weight) +
         weights[1]*bce_dice(out['aux8'], target, pos_weight) +
         weights[2]*bce_dice(out['aux16'], target, pos_weight))
    if q is not None and w_gate > 0 and out['gates'][0].requires_grad:
        l = l + w_gate*gate_loss(out['gates'], q)
    return l
