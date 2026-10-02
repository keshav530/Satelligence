"""RRDB generator scaffold (ESRGAN-style). UNTRAINED. The Swin block, discriminator and
GAN loop are not written yet. Kept importable without torch so the numpy pipeline still runs."""
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    class RDB(nn.Module):
        def __init__(self, nf=32, gc=16):
            super().__init__()
            self.c = nn.ModuleList([nn.Conv2d(nf + i * gc, gc if i < 4 else nf, 3, padding=1) for i in range(5)])
            self.act = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            f = [x]
            for i, conv in enumerate(self.c):
                y = conv(torch.cat(f, 1))
                if i < 4:
                    f.append(self.act(y))
            return x + 0.2 * y

    class RRDB(nn.Module):
        def __init__(self, nf=32):
            super().__init__()
            self.b = nn.Sequential(RDB(nf), RDB(nf), RDB(nf))

        def forward(self, x):
            return x + 0.2 * self.b(x)

    class Generator(nn.Module):
        """4-band in, 4-band out, x4 upscale. p_drop > 0 enables MC-dropout sampling."""
        def __init__(self, in_ch=4, nf=32, nb=4, scale=4, p_drop=0.1):
            super().__init__()
            self.head = nn.Conv2d(in_ch, nf, 3, padding=1)
            self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
            self.drop = nn.Dropout2d(p_drop)
            up = []
            for _ in range(scale // 2):
                up += [nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)]
            self.up = nn.Sequential(*up)
            self.tail = nn.Conv2d(nf, in_ch, 3, padding=1)

        def forward(self, x):
            f = self.head(x)
            f = f + self.drop(self.body(f))
            return self.tail(self.up(f))

    @torch.no_grad()
    def mc_dropout(model, x, n=8):
        """Per-pixel mean and std over n stochastic passes. std is the (uncalibrated) confidence signal."""
        model.eval()
        model.drop.train()
        outs = torch.stack([model(x) for _ in range(n)])
        return outs.mean(0), outs.std(0)
