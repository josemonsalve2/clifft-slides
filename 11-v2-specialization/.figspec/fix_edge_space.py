#!/usr/bin/env python3
"""Preserve leading/trailing spaces in <text> runs.

SVG inherits XML's default whitespace handling, so a <text> whose content
starts or ends with a space renders without it. That is invisible in the usual
case, but after fix_tspan.py splits a sentence into adjacent runs the word
break lives exactly at a run boundary -- so "knows " + "k" prints as "knowsk".
Converting only the edge spaces to U+00A0 restores the gap without touching
interior spacing (which should still collapse normally).
"""
import re, sys, os

def fix(path):
    src = open(path, encoding='utf-8').read()
    n = 0
    def repl(m):
        nonlocal n
        a, body = m.group(1), m.group(2)
        if '<' in body:
            return m.group(0)
        new = re.sub(r'^( +)', lambda x: '&#160;' * len(x.group(1)), body)
        new = re.sub(r'( +)$', lambda x: '&#160;' * len(x.group(1)), new)
        if new != body:
            n += 1
            return f'<text{a}>{new}</text>'
        return m.group(0)
    out = re.sub(r'<text\b([^>]*)>(.*?)</text>', repl, src, flags=re.S)
    if n:
        open(path, 'w', encoding='utf-8').write(out)
    print(f'{os.path.basename(path)}: preserved {n}')

for p in sys.argv[1:]:
    fix(p)
