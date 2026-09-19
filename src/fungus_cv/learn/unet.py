"""U-Net with a pretrained ResNet encoder, built only from torch/torchvision."""

from __future__ import annotations

from fungus_cv.segment.torch_device import import_torch

torch = import_torch()
nn = torch.nn
F = torch.nn.functional

ENCODERS = {"resnet18": (64, 64, 128, 256, 512), "resnet34": (64, 64, 128, 256, 512)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DIVISOR = 32  # input height/width must be multiples of this


def _use_certifi_if_needed() -> None:
    """python.org Python on macOS ships without root certificates, which breaks torchvision's
    weight download. Use certifi's bundle (installed alongside transformers) if present."""
    import os

    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def _conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _conv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    """Outputs one logit per pixel and class at full input resolution (``classes`` channels;
    classes may overlap, so each is its own yes/no).

    A small full-resolution branch feeds the last decoder stage so mask edges are not
    limited to the encoder's /2 resolution.
    """

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True, classes: int = 1):
        super().__init__()
        import torchvision.models as tvm

        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; choose from {sorted(ENCODERS)}")
        weights = None
        if pretrained:
            weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
                       "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1}[encoder]
        if weights is not None:
            _use_certifi_if_needed()
        try:
            enc = getattr(tvm, encoder)(weights=weights)
        except OSError as exc:  # URLError is an OSError
            raise RuntimeError(
                f"could not download ImageNet weights for {encoder}: {exc}. Check the internet "
                "connection (on macOS with python.org Python, run 'Install Certificates.command'), "
                "or train with --no-pretrained."
            ) from exc
        c0, c1, c2, c3, c4 = ENCODERS[encoder]
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2
        self.pool = enc.maxpool  # /4
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            enc.layer1, enc.layer2, enc.layer3, enc.layer4)  # /4 /8 /16 /32
        self.full_res = _conv(3, 16)
        self.up4 = _Up(c4, c3, 256)
        self.up3 = _Up(256, c2, 128)
        self.up2 = _Up(128, c1, 64)
        self.up1 = _Up(64, c0, 32)
        self.up0 = _Up(32, 16, 16)
        self.head = nn.Conv2d(16, classes, 1)

    def forward(self, x):
        full = self.full_res(x)
        s = self.stem(x)
        e1 = self.layer1(self.pool(s))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d = self.up4(e4, e3)
        d = self.up3(d, e2)
        d = self.up2(d, e1)
        d = self.up1(d, s)
        d = self.up0(d, full)
        return self.head(d)
