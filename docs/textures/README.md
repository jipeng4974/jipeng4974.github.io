# Texture sources

Self-hosted 512×512 tiles for the optional 玉笺 (classic) color mode. Both
sources are CC0.

| File | Source | License |
|---|---|---|
| `paper001-512.jpg` | [ambientCG Paper001](https://ambientcg.com/a/Paper001) | CC0 |
| `white-jade-512.jpg` | [ambientCG Onyx015](https://ambientcg.com/a/Onyx015) | CC0 |
| `white-jade-soft.png` | derived from `white-jade-512.jpg` | CC0 |

`paper001-512.jpg` is the original ×1.00 paper colour map, centre-cropped and
resized from the 1K source. It is painted on `body` only in classic mode; the
default light theme uses the flat `--bs-body-bg` surface.

`white-jade-512.jpg` cools and lightens the Onyx015 colour map for the classic
sidebar, portrait `.td-shell-subnav` topbar, footer and code-block surfaces: a
low-contrast milky tile with the amber cast pulled out, then washed with the
cool ivory base (`#f3f6f3` / `#f8fbf8`) in CSS so the onyx veining stays subtle
behind the post-title tree and code text.

`white-jade-soft.png` is the same tile with a flat 36% alpha, used by the
portrait contents sheet so the opened 目录 keeps the jade material while the
page still reads through it. All original materials are seamless.
