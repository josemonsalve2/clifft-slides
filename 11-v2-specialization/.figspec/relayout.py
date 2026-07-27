#!/usr/bin/env python3
"""Re-space text runs that fix_tspan.py laid out with the old flat width factor.

fix_tspan split each anchored <text> with inline tspans into sibling runs, but
positioned them with a flat 0.55 em/char estimate. Arial lowercase prose is
really ~0.42-0.47 em, so the runs ended up with visible gaps between them.

This re-lays each group contiguously using the measured per-character table,
preserving the group's original centre: fix_tspan placed the first run at
centre - total_old/2, so centre = x_first + total_old/2 is recoverable exactly.
"""
import re, sys, os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_WD = json.load(open(os.path.join(_HERE, 'widths.json')))
_FALLBACK = {'sans': 0.50, 'sans_bold': 0.52, 'mono': 0.60, 'dejavu': 0.55}
OLD_SANS, OLD_MONO = 0.55, 0.60

def font_of(fam, weight):
    f = (fam or '').lower()
    if 'monospace' in f:
        return 'mono'
    if 'dejavu' in f:
        return 'dejavu'
    return 'sans_bold' if str(weight) in ('700', '800', '900', 'bold') else 'sans'

def measure(text, size, font):
    t = _WD[font]; fb = _FALLBACK[font]
    return sum(t.get(c, fb) for c in text) * size

def esc_len(s):
    return len(re.sub(r'&(#\d+|[a-z]+);', 'X', s))

def attr(s, n, d=None):
    m = re.search(r'\b' + n + r'="([^"]*)"', s)
    return m.group(1) if m else d

TEXT = re.compile(r'<text\b([^>]*)>([^<]*)</text>')

def relayout(path):
    src = open(path, encoding='utf-8').read()
    groups, cur = [], []
    for m in TEXT.finditer(src):
        if cur and m.start() == cur[-1].end():          # directly adjacent
            if attr(m.group(1), 'y') == attr(cur[-1].group(1), 'y'):
                cur.append(m); continue
        if len(cur) > 1:
            groups.append(cur)
        cur = [m]
    if len(cur) > 1:
        groups.append(cur)

    edits, n = [], 0
    for g in groups:
        runs = []
        for m in g:
            a, txt = m.group(1), m.group(2)
            size = float(attr(a, 'font-size', '12'))
            fam  = attr(a, 'font-family', '')
            wt   = attr(a, 'font-weight', '400')
            runs.append((m, txt, size, font_of(fam, wt), fam))
        old = sum(esc_len(t) * sz * (OLD_MONO if f == 'mono' else OLD_SANS)
                  for _, t, sz, f, _ in runs)
        x0 = float(attr(g[0].group(1), 'x', '0'))
        centre = x0 + old / 2
        new_total = sum(measure(t, sz, f) for _, t, sz, f, _ in runs)
        x = centre - new_total / 2
        for m, t, sz, f, _ in runs:
            newa = re.sub(r'\bx="[^"]*"', f'x="{x:.1f}"', m.group(1), count=1)
            edits.append((m.start(), m.end(), f'<text{newa}>{t}</text>'))
            x += measure(t, sz, f)
        n += 1

    for s, e, rep in reversed(edits):
        src = src[:s] + rep + src[e:]
    if n:
        open(path, 'w', encoding='utf-8').write(src)
    print(f'{os.path.basename(path)}: relaid {n} group(s)')

for p in sys.argv[1:]:
    relayout(p)
