#!/usr/bin/env python3
"""
make-seal.py — generate a four-character Chinese seal (白文印, intaglio).

Pipeline:
  1. render each character with a seal-script font (中山王篆 by default),
     laying the four glyphs out in the traditional reading order — right column
     top→bottom, then left column top→bottom;
  2. thicken the hairlines into the fat strokes a carved stone seal has;
  3. weather the whole plate with a randomised multi-octave noise field:
     strokes get eaten from their edges inward, the plate's rim chips, and the
     ink picks up pinholes — every run differs unless --seed is pinned;
  4. punch the glyphs out of the plate, so the seal is red ink around
     background-coloured characters (the classic 白文 look).

Characters the font lacks are taken from a compatibility table (the font maps
the simplified code points of 時/鴻/遊/戲), and 昧 — absent from the font — is
assembled from its two components, 日 + 未.

Usage:
    python3 make-seal.py "時俗工巧" --out seal.png
    python3 make-seal.py 遊戲三昧 --out seal.png --seed 7 --thicken 14

Needs: pillow, numpy (see --help for all knobs).
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_FONT = os.path.expanduser("~/.local/share/fonts/JFZSKSealScript.ttf")

# The font only maps some code points: fall back to the simplified twin, whose
# seal-script glyph is the same shape.
FALLBACK = {"時": "时", "鴻": "鸿", "遊": "游", "戲": "戏"}

# Characters the font does not carry at all, assembled from components.
COMPOUND = {"昧": ("日", "未")}  # left + right

READING_ORDER = ["top-right", "bottom-right", "top-left", "bottom-left"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a float array in [0, 1] through PIL."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L")
    img = img.filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(img, dtype=np.float32) / 255.0


def fractal_field(rng: np.random.Generator, shape: tuple[int, int],
                  octaves=((70, 1.0), (26, 0.6), (9, 0.34), (3, 0.16))) -> np.ndarray:
    """Multi-octave value noise in [0, 1].

    Stone is not uniformly worn: it erodes in patches at several scales. Stacking
    blurred white noise at decreasing radii and weights approximates that, and
    costs nothing next to a real Perlin implementation. The radii are sized for
    chips and flakes — a fine-grained field would read as sandpaper instead of a
    chipped stone edge.
    """
    field = np.zeros(shape, np.float32)
    for sigma, amp in octaves:
        n = rng.random(shape).astype(np.float32)
        n = blur(n, sigma)
        n -= n.mean()
        std = n.std()
        if std > 1e-6:
            n /= std
        field += amp * n
    field -= field.min()
    span = field.max()
    return field / span if span > 1e-6 else np.zeros(shape, np.float32)


def to_rank(field: np.ndarray) -> np.ndarray:
    """Map a field to its percentile in [0, 1], so thresholds become wear ratios.

    An absolute threshold on a noise field is impossible to reason about — the
    same number eats 10% of one field and 60% of the next. Working in ranks means
    "wear away 18% of this" is exactly what happens.
    """
    flat = field.ravel()
    order = flat.argsort().argsort().astype(np.float32)
    return (order / max(1, flat.size - 1)).reshape(field.shape)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with an octagonal structuring element.

    numpy shifts instead of PIL's MaxFilter: the filters are O(size²) per pixel
    and a 12px radius on a 2400px plate is slow enough to notice, while eight
    shifted ORs stay linear.
    """
    for _ in range(max(0, radius)):
        m = mask.copy()
        m[1:, :] |= mask[:-1, :]
        m[:-1, :] |= mask[1:, :]
        m[:, 1:] |= mask[:, :-1]
        m[:, :-1] |= mask[:, 1:]
        m[1:, 1:] |= mask[:-1, :-1]
        m[:-1, :-1] |= mask[1:, 1:]
        m[1:, :-1] |= mask[:-1, 1:]
        m[:-1, 1:] |= mask[1:, :-1]
        mask = m
    return mask


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return ~dilate(~mask, radius)


def masks_from_image(path: str, plate_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Split an already-rendered seal into its plate and its characters.

    Sources like Kimi's come as red-on-white with *white* characters, so the
    glyphs and the paper share a colour and cannot be told apart by colour alone.
    Flood-filling the paper from the corners settles it: whatever the fill cannot
    reach is inside the plate, and the pale pixels in there are the characters.
    """
    im = Image.open(path).convert("RGB").resize((plate_px, plate_px), Image.LANCZOS)
    filled = im.copy()
    ImageDraw.floodfill(filled, (0, 0), (0, 255, 0), thresh=32)
    a = np.asarray(filled)
    paper = (a[..., 1] > 200) & (a[..., 0] < 110) & (a[..., 2] < 110)
    plate = ~paper

    rgb = np.asarray(im)
    pale = (rgb[..., 0] > 170) & (rgb[..., 1] > 170) & (rgb[..., 2] > 170)
    return plate, plate & pale


def rounded_plate(size: int, radius: int) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).rounded_rectangle([0, 0, size - 1, size - 1],
                                          radius=radius, fill=255)
    return np.asarray(img) > 127


# --------------------------------------------------------------------------- #
# glyph rendering
# --------------------------------------------------------------------------- #

def render_char(font_path: str, char: str, box: int, rng: np.random.Generator,
                squeeze: float = 1.0) -> np.ndarray:
    """Render one character, trimmed to its ink and centred in a box×box tile.

    `squeeze` scales the glyph horizontally: the 中山王篆 glyphs are tall and
    narrow, which leaves a square seal looking empty, so a gentle stretch fills
    the tile without distorting the script the way a general scale would.
    """
    probe = box * 3
    font = ImageFont.truetype(font_path, probe)
    canvas = Image.new("L", (probe * 2, probe * 2), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((probe // 2, probe // 2), char, font=font, fill=255)

    arr = np.asarray(canvas) > 40
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        raise SystemExit(f"font produced no ink for {char!r}")
    glyph = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = glyph.shape
    scale = (box * squeeze) / w
    if h * scale > box:                      # keep it inside the tile
        scale = box / h
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    glyph_img = Image.fromarray((glyph * 255).astype(np.uint8), "L")
    glyph_img = glyph_img.resize((new_w, new_h), Image.LANCZOS)

    tile = Image.new("L", (box, box), 0)
    tile.paste(glyph_img, ((box - new_w) // 2, (box - new_h) // 2))
    return np.asarray(tile) > 127


def render_composite(font_path: str, char: str, box: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Assemble a left/right compound character (昧 = 日 + 未)."""
    left, right = COMPOUND[char]
    tile = np.zeros((box, box), bool)
    half = int(box * 0.52)
    l = render_char(font_path, left, int(box * 0.86), rng, squeeze=0.9)
    r = render_char(font_path, right, int(box * 0.92), rng, squeeze=0.9)
    for part, offset in ((l, 0), (r, box - half)):
        ys, xs = np.nonzero(part)
        if len(xs) == 0:
            continue
        sub = part[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ph, pw = sub.shape
        scale = min(half / pw, (box * 0.92) / ph)
        sub_img = Image.fromarray((sub * 255).astype(np.uint8), "L").resize(
            (max(1, int(pw * scale)), max(1, int(ph * scale))), Image.LANCZOS)
        arr = np.asarray(sub_img) > 127
        y = (box - arr.shape[0]) // 2
        x = offset + (half - arr.shape[1]) // 2
        tile[y:y + arr.shape[0], x:x + arr.shape[1]] |= arr
    return tile


def layout_glyphs(text: str, plate: int, font_path: str,
                  rng: np.random.Generator, squeeze: float,
                  fill: float = 0.88) -> np.ndarray:
    """Place four glyphs in the traditional order: right column, then left.

    `fill` is how much of its cell a glyph occupies. Seals breathe through the
    red left between characters (分朱布白); letting the glyphs fill the cell edge
    to edge is what makes a generated seal look like a logo instead of a carving.
    """
    margin = int(plate * 0.08)
    gap = int(plate * 0.022)
    cell = (plate - 2 * margin - gap) // 2
    inner = int(cell * fill)
    offset = (cell - inner) // 2
    mask = np.zeros((plate, plate), bool)

    positions = [
        (margin + cell + gap, margin),               # top-right
        (margin + cell + gap, margin + cell + gap),  # bottom-right
        (margin, margin),                            # top-left
        (margin, margin + cell + gap),               # bottom-left
    ]
    for char, (x, y) in zip(text, positions):
        if char in COMPOUND:
            # The compound routine lays its parts out over the whole cell.
            tile = render_composite(font_path, char, inner, rng)
        else:
            glyph = FALLBACK.get(char, char)
            tile = render_char(font_path, glyph, inner, rng, squeeze=squeeze)
        y0, x0 = y + offset, x + offset
        mask[y0:y0 + inner, x0:x0 + inner] |= tile
    return mask


# --------------------------------------------------------------------------- #
# weathering
# --------------------------------------------------------------------------- #

def rough_edge(mask: np.ndarray, rng: np.random.Generator,
               strength: float = 0.6, sigma: float = 7.0) -> np.ndarray:
    """Displace the outline with noise so lines are not uniformly thick.

    Blur the mask into a soft ramp, add a noise field, threshold again: the edge
    lands wherever the two cross, which shifts it back and forth along the
    contour. That is what a chisel does — the groove runs deeper where the hand
    pushed — and it is the difference between a font sample and a carving.
    """
    soft = blur(mask.astype(np.float32), sigma)
    n = fractal_field(rng, mask.shape, octaves=((48, 1.0), (16, 0.5), (5, 0.25)))
    return (soft + (n - 0.5) * strength) > 0.5


def weather_strokes(strokes: np.ndarray, rng: np.random.Generator,
                    strength: float) -> np.ndarray:
    """Eat the strokes from their edges inward.

    A carved stroke loses its rim first: the chisel's edge crumbles while the
    middle of the groove survives. So the wear ratio is raised along the stroke
    outline, which makes the erosion follow the contours instead of blowing
    random holes through the middle of a line.
    """
    # The rim is taken from *inside* the stroke: `dilate & ~strokes` would mark
    # the empty moat around it, where erosion has nothing to eat.
    outline = strokes & ~erode(strokes, 7)
    rank = to_rank(fractal_field(rng, strokes.shape))
    wear = np.full(strokes.shape, 0.02 * strength, np.float32)
    wear[outline] += 0.45 * strength
    return strokes & ~(rank < wear)


def weather_plate(plate: np.ndarray, rng: np.random.Generator,
                  strength: float) -> np.ndarray:
    """Chip the rim of the plate: knocks, nibbles and the odd pinhole."""
    rim = dilate(plate, 12) & ~erode(plate, 12)
    # Low-frequency dominant noise: the rim should lose a few chunks, not turn
    # into an even furry edge all the way round.
    rank = to_rank(fractal_field(rng, plate.shape,
                                 octaves=((150, 1.0), (45, 0.5), (12, 0.2))))
    wear = np.full(plate.shape, 0.004 * strength, np.float32)
    wear[rim] += 0.34 * strength
    plate = plate & ~(rank < wear)

    # Pinholes: ink that never reached the paper.
    holes = rng.random(plate.shape) < (0.00035 * strength)
    holes = dilate(holes, 2)
    return plate & ~holes


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build(text: str, *, font_path: str, size: int, ink: tuple[int, int, int],
          seed: int | None, thicken: int, squeeze: float, wear: float,
          radius_ratio: float, work: int, fill: float,
          source_image: str | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    plate_px = work
    if source_image:
        plate, strokes = masks_from_image(source_image, plate_px)
        tip = "image"
    else:
        plate, strokes = None, None
        tip = "font"
    # --thicken is quoted in pixels of a 1200px seal, independent of --size and
    # --work: scaling it by work/size silently multiplied the stroke weight by
    # 6.7x when a small output was requested, welding the glyphs together.
    thicken_px = max(1, int(round(thicken * plate_px / 1200)))

    if tip == "font":
        strokes = layout_glyphs(text, plate_px, font_path, rng, squeeze, fill=fill)
    if thicken_px:
        strokes = dilate(strokes, thicken_px)
    # The blur has to stay well under half the stroke width, or the threshold
    # step dissolves the strokes instead of roughening their edges.
    strokes = rough_edge(strokes, rng, strength=0.45,
                         sigma=max(2.5, min(thicken_px * 0.7, plate_px * 0.005)))
    strokes = weather_strokes(strokes, rng, wear)

    if plate is None:
        plate = rounded_plate(plate_px, int(plate_px * radius_ratio))
    plate = weather_plate(plate, rng, wear)

    ink_layer = plate & ~strokes

    out = np.zeros((plate_px, plate_px, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = ink
    out[..., 3] = ink_layer.astype(np.uint8) * 255

    img = Image.fromarray(out, "RGBA").resize((size, size), Image.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="four characters, e.g. 時俗工巧")
    ap.add_argument("--from-image", help="weather an existing seal image instead of "
                                        "rendering text (its plate and glyphs are "
                                        "separated automatically)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--font", default=DEFAULT_FONT)
    ap.add_argument("--size", type=int, default=1200, help="output pixels (default 1200)")
    ap.add_argument("--work", type=int, default=2400, help="working pixels before downscale")
    ap.add_argument("--ink", default="#b3261e", help="seal ink colour (default cinnabar)")
    ap.add_argument("--seed", type=int, default=None, help="pin the randomness")
    ap.add_argument("--thicken", type=int, default=13, help="stroke fattening in output px")
    ap.add_argument("--squeeze", type=float, default=1.12, help="horizontal glyph stretch")
    ap.add_argument("--wear", type=float, default=1.0, help="weathering strength")
    ap.add_argument("--fill", type=float, default=0.88,
                    help="fraction of each cell a glyph fills (red between the characters)")
    ap.add_argument("--radius", type=float, default=0.05, help="corner radius, fraction of size")
    ap.add_argument("--preview", help="also write a white-background preview PNG")
    args = ap.parse_args()

    if bool(args.text) == bool(args.from_image):
        raise SystemExit("give either four characters or --from-image")
    if args.text and len(args.text) != 4:
        raise SystemExit("expected exactly four characters")

    ink = args.ink.lstrip("#")
    ink_rgb = tuple(int(ink[i:i + 2], 16) for i in (0, 2, 4))

    img = build(args.text or "", font_path=args.font, size=args.size, ink=ink_rgb,
                seed=args.seed, thicken=args.thicken, squeeze=args.squeeze,
                wear=args.wear, radius_ratio=args.radius, work=args.work,
                fill=args.fill, source_image=args.from_image)
    img.save(args.out)
    print(f"wrote {args.out}  ({img.width}×{img.height})")

    if args.preview:
        bg = Image.new("RGBA", img.size, (247, 241, 229, 255))
        bg.alpha_composite(img)
        bg.convert("RGB").save(args.preview)
        print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
