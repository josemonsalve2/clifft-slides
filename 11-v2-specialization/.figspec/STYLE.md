# Diagram house style — deck 11 (`11-v2-specialization/diagrams/`)

Every figure is a **hand-written, standalone SVG** matching the ten diagrams
already in `diagrams/`. Read `diagrams/memory-hierarchy-tiers.svg` and
`diagrams/persistent-kernel.svg` before writing anything — they are the
reference for structure, and you must match them.

## Hard requirements

- Root element: `<svg xmlns="http://www.w3.org/2000/svg" width="W" height="H"
  viewBox="0 0 W H" role="img" aria-labelledby="title desc">` followed
  immediately by `<title id="title">` and `<desc id="desc">`.
- **No external references.** No `<image>`, no web fonts, no CSS `@import`, no
  scripts. Font families are literal: `Arial, sans-serif` for prose,
  `monospace` for code/identifiers. The file must render standalone in a
  browser and inside a `<img src=...>` tag (so no inline `<style>` reliance on
  the host document).
- Background: a full-bleed `<rect width="W" height="H" fill="#1a1a2e"/>` as the
  first painted element.
- Canvas: 800–1100 wide, 420–620 tall. Pick per figure; wider for timelines,
  taller for stacked pipelines.
- **Nothing may overflow the viewBox and no two text elements may overlap.**
  Compute your own layout arithmetic; do not eyeball. Text is not measured by
  the browser here, so budget ~0.55 × font-size per character for
  `Arial` width and ~0.6 × for `monospace`, and keep boxes wider than that.
- Every number that appears in a figure must come from the DATA block of that
  figure's spec, verbatim. **Do not invent, round, or interpolate a number.**
  If the spec gives no number for something, draw it qualitatively with no
  number rather than making one up.

## Palette (from the deck's CSS variables — use these exact hex values)

| role | hex |
|---|---|
| background | `#1a1a2e` |
| panel / inset fill | `#0d0d1a` at 0.72 opacity |
| primary text | `#ffffff` |
| secondary text | `#dfe3ed` |
| tertiary / label text | `#aeb4c7` |
| dim / caption text | `#9ba3b8` |
| accent (deck brand, use for the "answer"/V2 side) | `#e94560` |
| highlight (cyan, use for callouts and arrows) | `#53d8fb` |
| good / win / fast | `#00cc66` (bright `#00ff80`) |
| warning / middle tier | `#ff8800` (bright `#ffb35c`) |
| bad / slow / V1 | `#cc2222` |
| neutral rule | `#ffffff` at 0.06–0.10 opacity |

Convention used throughout the deck and which you must keep:
**green = register tier / fast / correct**, **orange = coop tier / caution**,
**red = global tier / V1 / the bug**, **cyan = the explanatory callout**,
**`#e94560` = V2's answer**.

## Composition rules

1. Title at `y=31`, 21 px, bold, `#ffffff`, `text-anchor="middle"`, centred.
   Subtitle directly under at `y=50`, 11 px, `#aeb4c7`. Both required.
2. Use `<defs>` for gradients, arrow markers and glow filters — copy the
   idioms from `memory-hierarchy-tiers.svg` rather than inventing new ones.
3. Group related elements in `<g>` with a preceding XML comment naming the
   group (`<!-- Tier 2 -->`). The existing files do this and it is how they
   stay editable.
4. Prefer a **left→right "before/after"** or **top→bottom "stage"** layout with
   an explicit arrow between halves. Label the arrow with the effect
   (`1,883× fewer lines`), do not leave it bare.
5. When a figure carries a punchline, put it in a bottom band: a rounded rect
   in the accent colour at 0.10–0.18 fill opacity with a 1-px stroke, and one
   line of 12–13 px text. Reserve ~55 px of canvas height for it.
6. Log scales must be **labelled as log**, with visible tick values.
7. Legends go top-right or bottom-left, never over data.

## Style to avoid

- No drop shadows other than the `feGaussianBlur` glow idiom already used.
- No gratuitous 3-D, no skeuomorphism, no clip art.
- No text below 9 px.
- **Never put a `<tspan>` inside a `<text>` that has `text-anchor="middle"`
  or `text-anchor="end"`.** Browsers handle it, but cairosvg and ImageMagick
  both mis-position the tspan and the words overprint each other. If you need
  mixed weight in a centred line, either split it into two separate `<text>`
  elements at computed x positions, or use `text-anchor="start"` with an x you
  computed yourself.
- No pure `#000000` or `#ffffff` fills for large areas.
