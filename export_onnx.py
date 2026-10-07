"""Download official Real-ESRGAN compact weights, export to ONNX (dynamic H/W) and verify against PyTorch.
Run by CI; also locally:  pip install torch onnx onnxruntime && python scripts/export_onnx.py"""
import pathlib, urllib.request, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "site" / "models"; W = ROOT / "weights"
OUT.mkdir(parents=True, exist_ok=True); W.mkdir(exist_ok=True)
BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
CFG = {"realesr-general-x4v3": 32, "realesr-animevideov3": 16}

class SRVGGNetCompact(nn.Module):  # identical to realesrgan.archs.srvgg_arch.SRVGGNetCompact (prelu)
    def __init__(self, nf=64, nc=16, up=4):
        super().__init__(); self.up = up; self.body = nn.ModuleList([nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(nf)])
        for _ in range(nc): self.body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(nf)]
        self.body.append(nn.Conv2d(nf, 3 * up * up, 3, 1, 1)); self.upsampler = nn.PixelShuffle(up)
    def forward(self, x):
        o = x
        for l in self.body: o = l(o)
        return self.upsampler(o) + F.interpolate(x, scale_factor=self.up, mode="nearest")

import onnxruntime as ort
for name, nc in CFG.items():
    pth = W / f"{name}.pth"
    if not pth.exists(): urllib.request.urlretrieve(BASE + f"{name}.pth", pth)
    sd = torch.load(pth, map_location="cpu"); net = SRVGGNetCompact(nc=nc); net.load_state_dict(sd.get("params", sd)); net.eval()
    path = OUT / f"{name}.onnx"
    torch.onnx.export(net, torch.rand(1, 3, 64, 64), str(path), input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}})
    x = torch.rand(1, 3, 80, 56)  # different shape: proves dynamic axes + numerical equivalence
    ref = net(x).detach().numpy(); got = ort.InferenceSession(str(path)).run(None, {"input": x.numpy()})[0]
    d = float(np.abs(ref - got).max()); assert got.shape == (1, 3, 320, 224) and d < 1e-3, f"{name}: ONNX mismatch {d}"
    print(f"{name}: OK, max diff {d:.2e}, {path.stat().st_size/1e6:.1f} MB")
