#!/usr/bin/env python3
"""Static lint for the hand-written diagram SVGs.

Catches what browsers hide and cairosvg (our actual rasterizer) exposes:
  OVERFLOW      text escaping the viewBox
  OVERLAP       two runs printing on top of each other
  TSPAN-ANCHOR  an inline tspan inside a middle/end-anchored text -- cairosvg
                and ImageMagick both mis-position these
  GLYPH         a character the run's font does not carry; cairosvg does no
                per-glyph fallback, so it comes out as a tofu box

Font notes, measured (see .figspec/STYLE.md):
  "Arial"/sans  -> Liberation Sans here.  Lacks  U+21D2 U+2B07 U+26A0 U+2713
                   U+2717 U+2297.  Has ... arrows box-drawing minus bullets.
  monospace     -> DejaVu Sans Mono.  Carries everything we use.
  "DejaVu Sans" -> carries everything we use.
"""
import re, sys, glob, os, json

# Per-character advance widths, measured by rendering each glyph 30x through
# cairosvg and reading the ink span (.figspec/widths.json). A flat factor is
# not good enough: Arial lowercase prose averages 0.42 em while bold caps hit
# 0.58, so a single constant either cries wolf or misses real overlaps.
_WD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'widths.json')))
_FALLBACK = {'sans': 0.50, 'sans_bold': 0.52, 'mono': 0.60, 'dejavu': 0.55}

def pick_font(family, cls, weight):
    f = (family or '').lower()
    if 'monospace' in f or 'mono' in (cls or '').lower():
        return 'mono'
    if 'dejavu' in f:
        return 'dejavu'
    return 'sans_bold' if str(weight) in ('700', '800', '900', 'bold') else 'sans'

def measure(text, size, font):
    t = _WD[font]; fb = _FALLBACK[font]
    return sum(t.get(c, fb) for c in text) * size
# Characters Liberation Sans does NOT have. Safe in monospace / DejaVu Sans.
NARROW_FONT_MISSING = set('⇒⬇⚠✓✗✘✔⊗✕✖⇐⇑⇓')

def attr(s, name, default=None):
    m = re.search(r'\b' + re.escape(name) + r'="([^"]*)"', s)
    return m.group(1) if m else default

def unesc(s):
    s = re.sub(r'&#8203;|&#x200b;', '', s, flags=re.I)
    for a, b in (('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"')):
        s = s.replace(a, b)
    return re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), s)

def runs_of(svg):
    """Yield one record per rendered run, expanding explicitly-placed tspans.

    Tracks <g> nesting so inherited text-anchor / font-family / font-size are
    resolved the way a renderer resolves them -- without this, every element
    inside a `<g text-anchor="middle">` is measured as if left-anchored and
    reports bogus overflow.
    """
    INHERIT = ('text-anchor', 'font-family', 'font-size', 'font-weight')
    stack = [{'text-anchor': 'start', 'font-family': '', 'font-size': '12',
              'font-weight': '400'}]
    token = re.compile(r'<(/?)(g|text)\b([^>]*?)(/?)>(.*?)(?=<)', re.S)
    pos = 0
    for m in re.finditer(r'<(/?)(g|text)\b([^>]*?)(/?)>', svg, re.S):
        closing, tag, ta, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if tag == 'g':
            if closing:
                if len(stack) > 1:
                    stack.pop()
            elif not selfclose:
                env = dict(stack[-1])
                for k in INHERIT:
                    v = attr(ta, k)
                    if v is not None:
                        env[k] = v
                stack.append(env)
            continue
        if closing or selfclose:
            continue
        # <text ...> : grab its body
        close = svg.find('</text>', m.end())
        if close < 0:
            continue
        body = svg[m.end():close]
        if attr(ta, 'transform'):
            continue
        env = stack[-1]
        try:
            bx = float(attr(ta, 'x', '0')); by = float(attr(ta, 'y', '0'))
        except ValueError:
            continue
        size   = float(attr(ta, 'font-size', env['font-size']))
        anchor = attr(ta, 'text-anchor', env['text-anchor'])
        fam    = attr(ta, 'font-family', env['font-family']) or ''
        wt     = attr(ta, 'font-weight', env['font-weight']) or '400'
        cls    = attr(ta, 'class', '') or ''

        subs = list(re.finditer(r'<tspan\b([^>]*)>(.*?)</tspan>', body, re.S))
        placed = [x for x in subs if attr(x.group(1), 'x') or attr(x.group(1), 'dy')]
        if subs and len(placed) == len(subs):
            dy = 0.0
            for sN in subs:
                sx  = attr(sN.group(1), 'x')
                sdy = attr(sN.group(1), 'dy')
                if sdy:
                    dy += float(re.sub(r'[a-z]', '', sdy))
                txt = ' '.join(unesc(re.sub(r'<[^>]+>', '', sN.group(2))).split())
                if txt:
                    yield (float(sx) if sx else bx, by + dy, size, anchor,
                           attr(sN.group(1), 'font-family', fam) or fam, cls, txt,
                           False, attr(sN.group(1), 'font-weight', wt) or wt)
        else:
            inline = bool(subs) and anchor in ('middle', 'end')
            txt = ' '.join(unesc(re.sub(r'<[^>]+>', '', body)).split())
            if txt:
                yield bx, by, size, anchor, fam, cls, txt, inline, wt


def check(path):
    svg = open(path, encoding='utf-8').read()
    out = []
    vb = re.search(r'viewBox="([\d.\- ]+)"', svg)
    if not vb:
        return [f'{os.path.basename(path)}: no viewBox']
    _, _, VW, VH = [float(v) for v in vb.group(1).split()]

    boxes = []
    for x, y, size, anchor, fam, cls, s, inline, weight in runs_of(svg):
        if inline:
            out.append(f'TSPAN-ANCHOR: {s[:58]!r}')
        font = pick_font(fam, cls, weight)
        mono = font == 'mono'
        has_all = mono or font == 'dejavu'
        if not has_all:
            bad = sorted(NARROW_FONT_MISSING & set(s))
            if bad:
                out.append(f'GLYPH {"".join(bad)!r} (font={fam or "inherited"}) in {s[:40]!r}')
        w = measure(s, size, font)
        x0 = x - w / 2 if anchor == 'middle' else (x - w if anchor == 'end' else x)
        b = (x0, y - size * 0.78, x0 + w, y + size * 0.24)
        boxes.append((b, s))
        if b[0] < -2 or b[2] > VW + 2 or b[1] < -2 or b[3] > VH + 2:
            out.append(f'OVERFLOW: {s[:50]!r} -> x[{b[0]:.0f},{b[2]:.0f}] '
                       f'y[{b[1]:.0f},{b[3]:.0f}] vs {VW:.0f}x{VH:.0f}')

    for i in range(len(boxes)):
        (ax0, ay0, ax1, ay1), a = boxes[i]
        for j in range(i + 1, len(boxes)):
            (bx0, by0, bx1, by1), b = boxes[j]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > 3 and oy > 2:
                out.append(f'OVERLAP({ox:.0f}x{oy:.0f}): {a[:34]!r} vs {b[:34]!r}')
    return [f'{os.path.basename(path)}: {o}' for o in out]

if __name__ == '__main__':
    targets = sys.argv[1:] or sorted(glob.glob('diagrams/*.svg'))
    n = 0
    for t in targets:
        for line in check(t):
            print(line); n += 1
    print(f'--- {n} findings across {len(targets)} files')
