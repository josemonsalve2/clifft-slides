#!/usr/bin/env python3
"""Rasterize diagrams for review at a scale that does not invent defects.

Rendering at 1.2x and eyeballing the result is misleading: word spaces are ~3 px
at that scale, so anti-aliasing closes them and correct text reads as jammed.
Render at 4x and downsample with LANCZOS instead -- the glyphs stay separated
and what you see is what a reader sees.
"""
import sys, os
import cairosvg
from PIL import Image

OUT = '/tmp/figcheck'
os.makedirs(OUT, exist_ok=True)

for path in sys.argv[1:]:
    name = os.path.splitext(os.path.basename(path))[0]
    big = os.path.join(OUT, name + '.4x.png')
    cairosvg.svg2png(url=path, write_to=big, scale=4.0)
    im = Image.open(big)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(os.path.join(OUT, name + '.png'))
    print(f'{name}: {im.width}x{im.height} -> {im.width//2}x{im.height//2}')
