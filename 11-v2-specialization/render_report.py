#!/usr/bin/env python3
"""Render REPORT.md to a self-contained report.html styled like the deck.

Figures stay <img src="diagrams/*.svg">, so the HTML must be served from this
directory -- which is exactly where GitHub Pages serves it from.

    python3 render_report.py        # writes report.html

Only dependency beyond the stdlib is `markdown` (pip install --user markdown);
Pygments comes with it via the codehilite extension.
"""

import re
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

HERE = Path(__file__).parent
SRC = HERE / "REPORT.md"
DST = HERE / "report.html"

# Deck palette, verbatim from index.html's :root.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

:root {
  --bg-dark: #1a1a2e;
  --bg-card: #161b22;
  --accent: #e94560;
  --secondary: #0f3460;
  --highlight: #53d8fb;
  --text: #e0e0e0;
  --dim: #8888aa;
  --code-bg: #0d1117;
  --success: #4ec9b0;
  --warn: #e9a045;
  --purple: #c678dd;
  --border: #30363d;
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  margin: 0;
  background: var(--bg-dark);
  color: var(--text);
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 17px;
  line-height: 1.72;
}

/* ---- layout: fixed sidebar TOC + centred column ---- */
#toc {
  position: fixed;
  top: 0; left: 0; bottom: 0;
  width: 310px;
  overflow-y: auto;
  background: #14142a;
  border-right: 1px solid var(--border);
  padding: 1.4em 1.1em 3em;
  font-size: 0.8rem;
}
#toc .toc-title {
  color: var(--accent);
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.72rem;
  margin-bottom: 0.9em;
}
#toc ul { list-style: none; margin: 0; padding-left: 0.85em; }
#toc > ul { padding-left: 0; }
#toc li { margin: 0.14em 0; line-height: 1.4; }
#toc a { color: var(--dim); text-decoration: none; display: block; padding: 0.12em 0.3em; border-radius: 4px; }
#toc a:hover { color: var(--highlight); background: rgba(83,216,251,0.07); }
#toc > ul > li > a { color: var(--text); font-weight: 700; margin-top: 0.5em; }

#main { margin-left: 310px; padding: 3em 3.2em 8em; }
.wrap { max-width: 1020px; margin: 0 auto; }

@media (max-width: 1100px) {
  #toc { position: static; width: auto; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
  #main { margin-left: 0; padding: 2em 1.2em 5em; }
}

/* ---- headings ---- */
h1 { font-size: 2.35rem; color: var(--accent); font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; margin: 0 0 0.25em; }
h2 {
  font-size: 1.6rem; color: var(--highlight); font-weight: 700;
  border-bottom: 2px solid var(--accent); padding-bottom: 0.18em;
  margin: 2.4em 0 0.7em; letter-spacing: -0.01em; scroll-margin-top: 1em;
}
h3 { font-size: 1.18rem; color: var(--accent); font-weight: 700; margin: 1.9em 0 0.4em; scroll-margin-top: 1em; }
h4 { font-size: 1.02rem; color: var(--highlight); font-weight: 600; margin: 1.4em 0 0.3em; scroll-margin-top: 1em; }
h5, h6 { font-size: 0.95rem; color: var(--dim); font-weight: 700; margin: 1.2em 0 0.3em; }

p { margin: 0.85em 0; }
strong { color: var(--accent); font-weight: 700; }
em { color: var(--highlight); font-style: normal; }
a { color: var(--highlight); }
hr { border: none; border-top: 1px solid var(--border); margin: 2.4em 0; }

ul, ol { margin: 0.8em 0; padding-left: 1.5em; }
li { margin: 0.3em 0; }

/* ---- code ---- */
code {
  background: var(--bg-card); color: var(--highlight);
  padding: 0.13em 0.38em; border-radius: 4px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.84em;
}
pre {
  background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.95em 1.1em; margin: 1.1em 0; overflow-x: auto;
}
pre code, .codehilite pre {
  background: transparent; padding: 0; color: var(--text);
  font-size: 0.79rem; line-height: 1.62; display: block;
}
.codehilite { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; margin: 1.1em 0; overflow-x: auto; }
.codehilite pre { border: none; margin: 0; background: transparent; }

/* ---- tables ---- */
table { border-collapse: collapse; width: 100%; margin: 1.2em 0; font-size: 0.84rem; }
th { background: var(--secondary); color: var(--highlight); padding: 0.5em 0.75em; border: 1px solid var(--border); text-align: left; font-weight: 700; }
td { padding: 0.45em 0.75em; border: 1px solid var(--border); vertical-align: top; }
tbody tr:nth-child(even) { background: rgba(255,255,255,0.018); }

/* ---- blockquotes: the report uses them for callouts ---- */
blockquote {
  margin: 1.3em 0; padding: 0.9em 1.2em;
  background: rgba(233,69,96,0.07);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0;
}
blockquote > :first-child { margin-top: 0; }
blockquote > :last-child { margin-bottom: 0; }
blockquote h3, blockquote h4 { margin-top: 0.2em; }

/* ---- figures: same artwork the deck embeds ---- */
figure { margin: 1.8em auto; text-align: center; max-width: 100%; }
figure img {
  width: 100%; max-width: 900px; border-radius: 8px;
  display: block; margin: 0 auto;
  background: #1a1a2e;
}
figcaption {
  font-size: 0.79rem; color: var(--dim); line-height: 1.55;
  margin: 0.7em auto 0; max-width: 820px; text-align: left;
}
figcaption b { color: var(--highlight); }

/* ---- masthead ---- */
.masthead { border-bottom: 1px solid var(--border); padding-bottom: 1.4em; margin-bottom: 0.5em; }
.masthead .kicker { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim); margin-bottom: 0.6em; }
.masthead .links { margin-top: 1.1em; font-size: 0.82rem; }
.masthead .links a {
  display: inline-block; margin-right: 0.6em; padding: 0.35em 0.8em;
  border: 1px solid var(--border); border-radius: 6px;
  text-decoration: none; color: var(--highlight); background: var(--bg-card);
}
.masthead .links a:hover { border-color: var(--highlight); }
"""


def build_toc(html: str) -> str:
    """Nested TOC from the h2/h3 ids python-markdown's toc extension emitted."""
    heads = re.findall(r'<h([23]) id="([^"]+)">(.*?)</h[23]>', html, re.S)
    out, depth = [], 0
    for level, hid, text in heads:
        level = int(level)
        text = re.sub(r"<[^>]+>", "", text).strip()
        want = level - 2
        while depth < want:
            out.append("<ul>")
            depth += 1
        while depth > want:
            out.append("</ul>")
            depth -= 1
        out.append(f'<li><a href="#{hid}">{text}</a></li>')
    out += ["</ul>"] * depth
    return "<ul>" + "".join(out) + "</ul>"


def main() -> int:
    text = SRC.read_text()

    md = markdown.Markdown(
        extensions=["extra", "codehilite", "toc", "sane_lists", "admonition"],
        extension_configs={
            "codehilite": {"guess_lang": False, "noclasses": False},
            "toc": {"permalink": False},
        },
    )
    body = md.convert(text)

    # The first h1 becomes the masthead; the rest of the doc keeps its flow.
    pyg = HtmlFormatter(style="monokai").get_style_defs(".codehilite")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>From Interpreter to Specializer &mdash; the clifft GPU Backend</title>
<style>
{CSS}
{pyg}
.codehilite {{ background: var(--code-bg); }}
</style>
</head>
<body>
<nav id="toc">
  <div class="toc-title">Contents</div>
  {build_toc(body)}
</nav>
<div id="main"><div class="wrap">
<div class="masthead">
  <div class="kicker">Technical report &bull; clifft GPU backend</div>
  <div class="links">
    <a href="index.html">&#9654; Slide deck (274 slides)</a>
    <a href="../index.html">All decks</a>
    <a href="https://github.com/unitaryfoundation/clifft">clifft repo</a>
  </div>
</div>
{body}
</div></div>
</body>
</html>
"""
    DST.write_text(html)
    print(f"wrote {DST} ({len(html):,} B, {html.count('<figure>')} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
