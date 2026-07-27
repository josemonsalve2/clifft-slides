#!/usr/bin/env python3
"""Space adjacent text runs by measuring them, not by predicting them.

After fix_tspan.py splits an anchored <text> into sibling runs, the runs need
correct x offsets. A per-character width table gets close but drifts a few
pixels over a long sentence (kerning, bearings), and a few pixels is exactly
the difference between a word space and two words jammed together.

So measure instead: render each run on its own through the same cairosvg that
produces the report figures, read the ink extents, and chain the runs with one
real space between them. The result is exact by construction for this renderer.

Group left edges are preserved for start-anchored groups; for groups that were
originally centred, pass the centre through --keep-centre.
"""
import re, json, os, sys, subprocess, tempfile, html

import cairosvg
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(_HERE, 'widths.json')))
SCALE = 4.0
_cache = {}

def font_of(fam, wt):
    f = (fam or '').lower()
    if 'monospace' in f: return 'mono'
    if 'dejavu' in f:    return 'dejavu'
    return 'sans_bold' if str(wt) in ('700','800','900','bold') else 'sans'

def attr(s, n, d=None):
    m = re.search(r'\b' + n + r'="([^"]*)"', s)
    return m.group(1) if m else d

def unesc(s):
    return re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))),
                  s.replace('&amp;','&').replace('&lt;','<').replace('&gt;','>'))

def _inkspan(text, fam, size, weight, style):
    wattr = f'font-weight="{weight}" ' if weight and weight != '400' else ''
    sattr = f'font-style="{style}" ' if style else ''
    width = max(3000, int(len(text) * size * 1.4) + 400)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {width} {size*3:.0f}" width="{width}" height="{size*3:.0f}">'
           f'<rect width="{width}" height="{size*3:.0f}" fill="#000"/>'
           f'<text x="100" y="{size*2:.0f}" font-family="{fam}" font-size="{size}" '
           f'{wattr}{sattr}fill="#fff" xml:space="preserve">{html.escape(text)}</text></svg>')
    with tempfile.NamedTemporaryFile('w', suffix='.svg', delete=False) as f:
        f.write(svg); tmp = f.name
    png = tmp + '.png'
    cairosvg.svg2png(url=tmp, write_to=png, scale=SCALE)
    im = Image.open(png).convert('L'); Wp, Hp = im.size; px = im.load()
    xs = [x for x in range(Wp) if any(px[x, y] > 90 for y in range(Hp))]
    os.unlink(tmp); os.unlink(png)
    if not xs:
        return None
    return (min(xs) / SCALE, (max(xs) + 1) / SCALE)


def advance(text, fam, size, weight, style):
    """Exact advance width of `text`, in user units.

    Ink extents are not the advance -- they omit the side bearings, so chaining
    runs by ink edge silently eats a pixel or two at every boundary, which is
    most of a word space. Rendering the string twice and differencing the ink
    spans cancels both bearings exactly:  span(SS) - span(S) == advance(S).
    """
    key = (text, fam, size, weight, style)
    if key in _cache:
        return _cache[key]
    if not text:
        _cache[key] = 0.0
        return 0.0
    if not text.strip():
        _cache[key] = W[font_of(fam, weight)][' '] * size * len(text)
        return _cache[key]
    one = _inkspan(text, fam, size, weight, style)
    two = _inkspan(text + text, fam, size, weight, style)
    if one is None or two is None:
        _cache[key] = sum(W[font_of(fam, weight)].get(c, 0.5) for c in text) * size
    else:
        _cache[key] = (two[1] - two[0]) - (one[1] - one[0])
    return _cache[key]


TEXT = re.compile(r'<text\b([^>]*)>([^<]*)</text>')

def process(path):
    src = open(path, encoding='utf-8').read()
    groups, cur = [], []
    for m in TEXT.finditer(src):
        if cur and m.start() == cur[-1].end() and attr(m.group(1),'y') == attr(cur[-1].group(1),'y'):
            cur.append(m); continue
        if len(cur) > 1: groups.append(cur)
        cur = [m]
    if len(cur) > 1: groups.append(cur)

    edits = []
    for g in groups:
        runs = []
        for m in g:
            a = m.group(1)
            txt = unesc(m.group(2)).replace(' ', ' ')
            runs.append((m, a, txt,
                         attr(a, 'font-family', ''), float(attr(a, 'font-size', '12')),
                         attr(a, 'font-weight', '400'), attr(a, 'font-style', '')))

        # measured advance of each run's trimmed body, plus the space that
        # belonged to the boundary
        bodies = [r[2].strip() for r in runs]
        widths = [advance(body, fam, size, wt, st)
                  for (m, a, txt, fam, size, wt, st), body in zip(runs, bodies)]
        # Whether a word gap belongs at each boundary. Explicit edge whitespace
        # is authoritative when present, but earlier passes may already have
        # stripped it -- so fall back to the text itself: word char meeting word
        # char needs a space; a following ',' or ')' does not.
        NO_SPACE_BEFORE = ',.;:)]}!?'
        NO_SPACE_AFTER  = '([{'
        gaps = []
        for i, (m, a, txt, fam, size, wt, st) in enumerate(runs):
            if i + 1 >= len(runs):
                gaps.append(0.0); continue
            nxt = runs[i+1][2]
            if txt != txt.rstrip() or nxt != nxt.lstrip():
                want = True
            else:
                prev_c = txt.strip()[-1:] or ' '
                next_c = nxt.strip()[:1] or ' '
                want = (next_c not in NO_SPACE_BEFORE and prev_c not in NO_SPACE_AFTER)
            gaps.append(W[font_of(fam, wt)][' '] * size if want else 0.0)

        # Preserve the group's optical centre, not its left edge. fix_tspan
        # produced these groups from a middle-anchored <text>, so re-spacing
        # them left-aligned would drag every centred caption off-centre.
        # The old centre is recoverable from the current run positions: the
        # group spans x_first .. x_last + advance(last).
        total = sum(widths) + sum(gaps[:-1])
        x_first = float(attr(g[0].group(1), 'x', '0'))
        x_last  = float(attr(g[-1].group(1), 'x', '0'))
        old_span = (x_last + widths[-1]) - x_first
        x = x_first + (old_span - total) / 2.0
        for i, ((m, a, txt, fam, size, wt, st), body) in enumerate(zip(runs, bodies)):
            na = re.sub(r'\bx="[^"]*"', f'x="{x:.1f}"', a, count=1)
            edits.append((m.start(), m.end(),
                          f'<text{na}>{html.escape(body)}</text>'))
            x += widths[i] + gaps[i]

    for s, e, r in reversed(edits):
        src = src[:s] + r + src[e:]
    if edits:
        open(path, 'w', encoding='utf-8').write(src)
    print(f'{os.path.basename(path)}: respaced {len(groups)} group(s)')

for p in sys.argv[1:]:
    process(p)
