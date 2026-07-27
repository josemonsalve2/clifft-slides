#!/usr/bin/env python3
"""Retarget text runs containing check/cross glyphs at a font that has them.

cairosvg resolves "Arial" to Liberation Sans (metric-compatible) and does NOT
do per-glyph fallback, so U+2713/2717/2297 come out as tofu boxes. DejaVu Sans
carries all three. Rewriting just the affected runs keeps every other label on
the house font.
"""
import re, sys

NEED = set('✓✗✘✔⊗✕✖⇒⬇⚠⇐⇑⇓')

def fix(path):
    src = open(path, encoding='utf-8').read()
    n = 0
    def repl(m):
        nonlocal n
        attrs, body = m.group(1), m.group(2)
        plain = re.sub(r'<[^>]+>', '', body)
        if not (NEED & set(plain)):
            return m.group(0)
        if 'monospace' in attrs:            # monospace already renders them
            return m.group(0)
        n += 1
        if re.search(r'font-family="[^"]*"', attrs):
            attrs = re.sub(r'font-family="[^"]*"', 'font-family="DejaVu Sans"', attrs)
        else:
            attrs += ' font-family="DejaVu Sans"'
        return f'<text{attrs}>{body}</text>'
    new = re.sub(r'<text\b([^>]*)>(.*?)</text>', repl, src, flags=re.S)
    if n:
        open(path, 'w', encoding='utf-8').write(new)
    print(f'{path}: retargeted {n}')

for p in sys.argv[1:]:
    fix(p)
