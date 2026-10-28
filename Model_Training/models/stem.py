import torch, torch.nn as nn, torch.nn.functional as F

SSL4EO_S2_ORDER = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
OUR_S2_BANDS    = ['B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']   # 11


class SplitStem(nn.Module):
    """
    Two parallel first-layer convolutions whose outputs are summed:
        raw     : 11 real bands, initialised from SSL4EO
        derived : NDVI + NDBI,   ZERO-initialised  (Part 13 Q8)

    Keeping them separate is not cosmetic. It makes
      (a) excluding the derived columns from weight decay a one-line
          parameter-group selection, and
      (b) the 'with/without derived channels' ablation a boolean flag.
    """
    def __init__(self, n_raw=11, n_derived=2, out_ch=64, use_derived=True):
        super().__init__()
        self.use_derived = use_derived
        self.raw = nn.Conv2d(n_raw, out_ch, 7, stride=2, padding=3, bias=False)
        self.derived = nn.Conv2d(n_derived, out_ch, 7, stride=2, padding=3, bias=False)
        nn.init.zeros_(self.derived.weight)          # <-- the confirmed choice

    def forward(self, x_raw, x_derived):
        y = self.raw(x_raw)
        if self.use_derived:
            y = y + self.derived(x_derived)
        return y


@torch.no_grad()
def load_ssl4eo_s2(stem: SplitStem, ckpt_conv1: torch.Tensor):
    """
    ckpt_conv1 : [64, 13, 7, 7] from ssl4eo-resnet18-s2c-moco
    Keeps the 11 shared bands and REDISTRIBUTES the dropped B1/B10 columns
    so the stem's total response magnitude is preserved. Without this the
    first-layer activations shrink and early fine-tuning is unstable.
    """
    keep = [SSL4EO_S2_ORDER.index(b) for b in OUR_S2_BANDS]
    drop = [i for i in range(13) if i not in keep]              # B1, B10
    w = ckpt_conv1[:, keep].clone()
    w = w + ckpt_conv1[:, drop].sum(dim=1, keepdim=True) / len(keep)
    stem.raw.weight.copy_(w)
    return stem


@torch.no_grad()
def load_ssl4eo_s1(stem: SplitStem, ckpt_conv1: torch.Tensor):
    """ckpt_conv1 : [64, 2, 7, 7] (VV, VH). Third channel = cross-ratio, zero-init."""
    stem.raw.weight[:, :2].copy_(ckpt_conv1)
    stem.raw.weight[:, 2:].zero_()
    return stem


# ---------------- unit test that MUST pass before training ----------------
@torch.no_grad()
def verify_stem_surgery(ckpt_conv1, tol=0.98):
    """
    With derived channels zeroed, the adapted stem must reproduce the
    original 13-band stem on the shared bands. Cosine similarity > 0.98.
    """
    orig = nn.Conv2d(13, 64, 7, stride=2, padding=3, bias=False)
    orig.weight.copy_(ckpt_conv1)
    stem = load_ssl4eo_s2(SplitStem(), ckpt_conv1)

    x13 = torch.randn(2, 13, 128, 128)
    x13[:, [SSL4EO_S2_ORDER.index('B1'), SSL4EO_S2_ORDER.index('B10')]] = 0
    keep = [SSL4EO_S2_ORDER.index(b) for b in OUR_S2_BANDS]

    a = orig(x13).flatten()
    b = stem(x13[:, keep], torch.zeros(2, 2, 128, 128)).flatten()
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f'stem surgery cosine similarity = {cos:.4f}')
    assert cos > tol, 'STEM SURGERY FAILED -- check the band ordering'
    return cos
