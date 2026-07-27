#!/usr/bin/env python3
"""Flatten <tspan> runs out of anchored <text> elements.

cairosvg and ImageMagick both mis-position a tspan inside a text that carries
text-anchor="middle"/"end" -- the runs overprint. Browsers get it right, so the
SVG is not wrong, but our rasterizers are what produce the report figures.
The fix is mechanical: measure each run, then re-emit the whole line as
left-anchored sibling <text> elements at precomputed x offsets.

Tspans that already carry their own x= are the explicit multi-line pattern and
are left alone.
"""
import re, sys

W_SANS, W_MONO = 0.55, 0.60

def esc_len(s):
    """Character count with entities collapsed to one glyph."""
    return len(re.sub(r'&(#\d+|[a-z]+);', 'X', s))

def run_width(text, size, mono, bold):
    f = W_MONO if mono else W_SANS
    if bold and not mono:
        f *= 1.045                      # Arial Bold is a shade wider
    return esc_len(text) * size * f

def attr(s, name, default=None):
    m = re.search(r'\b' + name + r'="([^"]*)"', s)
    return m.group(1) if m else default

def flatten(svg):
    out, n = [], 0
    pos = 0
    for m in re.finditer(r'<text\b([^>]*)>(.*?)</text>', svg, re.S):
        attrs, body = m.group(1), m.group(2)
        anchor = attr(attrs, 'text-anchor', 'start')
        if anchor not in ('middle', 'end') or '<tspan' not in body:
            continue
        if re.search(r'<tspan[^>]*\bx=', body):     # explicit multi-line: leave it
            continue

        try:
            X = float(attr(attrs, 'x', '0'))
        except ValueError:
            continue
        size = float(attr(attrs, 'font-size', '12'))
        base_family = attr(attrs, 'font-family', 'Arial, sans-serif')
        base_fill   = attr(attrs, 'fill', '#ffffff')
        base_weight = attr(attrs, 'font-weight', '')
        base_style  = attr(attrs, 'font-style', '')
        carry = ''
        for k in ('y', 'filter', 'opacity', 'letter-spacing'):
            v = attr(attrs, k)
            if v is not None:
                carry += f' {k}="{v}"'

        # split body into (text, family, fill, weight, style) runs
        runs, i = [], 0
        for t in re.finditer(r'<tspan\b([^>]*)>(.*?)</tspan>', body, re.S):
            if t.start() > i:
                runs.append((body[i:t.start()], base_family, base_fill,
                             base_weight, base_style))
            ta = t.group(1)
            runs.append((t.group(2),
                         attr(ta, 'font-family', base_family),
                         attr(ta, 'fill', base_fill),
                         attr(ta, 'font-weight', base_weight),
                         attr(ta, 'font-style', base_style)))
            i = t.end()
        if i < len(body):
            runs.append((body[i:], base_family, base_fill, base_weight, base_style))
        runs = [r for r in runs if r[0] != '']
        if not runs:
            continue

        widths = [run_width(t, size, 'monospace' in fam, w in ('700', 'bold'))
                  for t, fam, _, w, _ in runs]
        total = sum(widths)
        x = X - total / 2 if anchor == 'middle' else X - total

        pieces = []
        for (t, fam, fill, weight, style), w in zip(runs, widths):
            a = (f'<text x="{x:.1f}"{carry} fill="{fill}" font-family="{fam}" '
                 f'font-size="{size:g}"')
            if weight: a += f' font-weight="{weight}"'
            if style:  a += f' font-style="{style}"'
            pieces.append(a + f'>{t}</text>')
            x += w
        out.append((m.start(), m.end(), ''.join(pieces)))
        n += 1

    for start, end, rep in reversed(out):
        svg = svg[:start] + rep + svg[end:]
    return svg, n

if __name__ == '__main__':
    for path in sys.argv[1:]:
        src = open(path, encoding='utf-8').read()
        new, n = flatten(src)
        if n:
            open(path, 'w', encoding='utf-8').write(new)
        print(f'{path}: flattened {n}')
