"""Planche contact : assemble plusieurs rendus en une grille legendee.

    bpyenv/bin/python scripts/contact_sheet.py --out renders/planche.png \
        --cols 3 renders/var_polished.png:POLISHED renders/var_gold.png:GOLD

Le texte est trace par une petite fonte bitmap 5x7 integree : cela evite de
dependre d'une police du systeme, qui n'existe pas forcement dans un conteneur.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# fonte 5x7, uniquement les caracteres utiles aux legendes
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("01110", "10000", "11110", "10001", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000",) * 7,
}


def draw_text(img, text, x, y, scale=3, color=(0.92, 0.94, 0.97)):
    for ch in text.upper():
        glyph = FONT.get(ch)
        if glyph is None:
            x += 6 * scale
            continue
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    y0, x0 = y + r * scale, x + c * scale
                    img[y0:y0 + scale, x0:x0 + scale, :3] = color
        x += 6 * scale
    return x


def load(path):
    import bpy
    im = bpy.data.images.load(os.path.abspath(path))
    w, h = im.size
    px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(im)
    return px


def save(px, path):
    import bpy
    h, w = px.shape[:2]
    out = bpy.data.images.new("planche", width=w, height=h, alpha=True)
    out.pixels = px[::-1].ravel().tolist()
    out.file_format = "PNG"
    out.filepath_raw = os.path.abspath(path)
    out.save()
    bpy.data.images.remove(out)


def build(entries, cols=3, gap=14, label_h=44, bg=0.055):
    tiles = [(load(p), lab) for p, lab in entries]
    th, tw = tiles[0][0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols

    W = cols * tw + (cols + 1) * gap
    H = rows * (th + label_h) + (rows + 1) * gap
    sheet = np.zeros((H, W, 4), dtype=np.float32)
    sheet[..., :3] = bg
    sheet[..., 3] = 1.0

    for i, (img, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + label_h + gap)
        sheet[y:y + th, x:x + tw, :3] = img[..., :3]
        draw_text(sheet, label, x + 4, y + th + 14, scale=3)
    return sheet


def main():
    argv = sys.argv[1:]
    out = "renders/planche.png"
    cols = 3
    entries = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            out = argv[i + 1]
            i += 2
        elif a == "--cols":
            cols = int(argv[i + 1])
            i += 2
        else:
            path, _, label = a.partition(":")
            entries.append((path, label or os.path.basename(path)))
            i += 1
    save(build(entries, cols=cols), out)
    print("ECRIT", out)


if __name__ == "__main__":
    main()
