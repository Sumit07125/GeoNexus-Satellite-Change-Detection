"""
conftest.py — pytest configuration for Geo-Nexus backend test suite.

Stubs out heavy optional dependencies (torch, earthengine-api, rasterio,
scipy, opencv) so the test suite can be imported and run without a full
ML environment. Tests that actually exercise these libraries are expected
to skip themselves or use their own mocks.
"""
import sys
import types


def _make_stub(name: str, **attrs) -> types.ModuleType:
    """Return a minimal stub module registered in sys.modules."""
    mod = types.ModuleType(name)
    mod.__spec__ = None  # prevents importlib from trying to reload it
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ── torch ──────────────────────────────────────────────────────────────────────
if "torch" not in sys.modules:
    torch_mod = _make_stub("torch")
    torch_mod.cuda = _make_stub("torch.cuda")
    torch_mod.cuda.is_available = lambda: False
    torch_mod.no_grad = lambda: __import__("contextlib").nullcontext()

    # torch.nn stub
    nn_mod = _make_stub("torch.nn")

    class _Module:
        def __init__(self, *a, **kw): pass
        def eval(self): return self
        def to(self, *a, **kw): return self
        def load_state_dict(self, *a, **kw): pass
        def parameters(self): return iter([])
        def __call__(self, *a, **kw): return None

    nn_mod.Module = _Module
    for sub in ("Sequential", "Conv2d", "BatchNorm2d", "ReLU", "Sigmoid",
                "Linear", "Dropout", "MaxPool2d", "AdaptiveAvgPool2d",
                "ConvTranspose2d", "LeakyReLU", "GELU", "LayerNorm",
                "MultiheadAttention", "TransformerEncoderLayer",
                "TransformerEncoder", "Embedding", "Upsample"):
        setattr(nn_mod, sub, _Module)
    torch_mod.nn = nn_mod
    _make_stub("torch.nn.functional")

    # torch.Tensor stub
    class _Tensor:
        def __init__(self, data=None): self.data = data
        def float(self): return self
        def unsqueeze(self, *a): return self
        def sigmoid(self): return self
        def cpu(self): return self
        def numpy(self): import numpy as np; return np.zeros((1, 1))

    torch_mod.Tensor = _Tensor
    torch_mod.tensor = lambda *a, **kw: _Tensor()
    torch_mod.from_numpy = lambda *a, **kw: _Tensor()
    torch_mod.load = lambda *a, **kw: {}
    torch_mod.device = lambda *a, **kw: "cpu"
    torch_mod.float32 = "float32"

    _make_stub("torch.utils")
    _make_stub("torch.utils.data")

# ── torchvision ────────────────────────────────────────────────────────────────
if "torchvision" not in sys.modules:
    tv_mod = _make_stub("torchvision")
    tv_models = _make_stub("torchvision.models")

    class _ResNet:
        def __init__(self, *a, **kw): pass
        def eval(self): return self
        def to(self, *a, **kw): return self
        def load_state_dict(self, *a, **kw): pass
        def parameters(self): return iter([])
        def __call__(self, *a, **kw): return None
        layer1 = layer2 = layer3 = layer4 = None
        fc = None

    tv_models.resnet18 = lambda *a, **kw: _ResNet()
    tv_models.resnet50 = lambda *a, **kw: _ResNet()
    tv_mod.models = tv_models
    _make_stub("torchvision.transforms")
    _make_stub("torchvision.transforms.functional")

# ── earthengine-api (ee) ───────────────────────────────────────────────────────
if "ee" not in sys.modules:
    ee_mod = _make_stub("ee")
    ee_mod.Initialize = lambda *a, **kw: None
    ee_mod.Image = object
    ee_mod.ImageCollection = lambda *a, **kw: object()

    class _Filter:
        @staticmethod
        def eq(*a, **kw): return None
        @staticmethod
        def listContains(*a, **kw): return None
        @staticmethod
        def date(*a, **kw): return None
        @staticmethod
        def bounds(*a, **kw): return None

    ee_mod.Filter = _Filter

    class _Join:
        @staticmethod
        def inner(): return _Join()
        def apply(self, *a, **kw): return object()

    ee_mod.Join = _Join
    ee_mod.Geometry = _make_stub("ee.Geometry")
    ee_mod.Geometry.Rectangle = lambda *a, **kw: object()
    ee_mod.ServiceAccountCredentials = lambda *a, **kw: object()
    _make_stub("ee.oauth")

# ── rasterio ───────────────────────────────────────────────────────────────────
if "rasterio" not in sys.modules:
    _make_stub("rasterio")
    _make_stub("rasterio.transform")
    _make_stub("rasterio.crs")

# ── scipy ──────────────────────────────────────────────────────────────────────
if "scipy" not in sys.modules:
    _make_stub("scipy")
    _make_stub("scipy.ndimage")
    _make_stub("scipy.stats")

# ── opencv (cv2) ───────────────────────────────────────────────────────────────
if "cv2" not in sys.modules:
    _make_stub("cv2")

# ── pystac_client / pystac ─────────────────────────────────────────────────────
if "pystac_client" not in sys.modules:
    _make_stub("pystac_client")
if "pystac" not in sys.modules:
    _make_stub("pystac")

# ── osgeo / gdal (sometimes pulled in by rasterio) ────────────────────────────
if "osgeo" not in sys.modules:
    _make_stub("osgeo")
    _make_stub("osgeo.gdal")
