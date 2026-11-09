"""Convert a path-only EMF (filled paths in black and white, such as a logo) into a vector PDF.

    python scripts/emf_to_pdf.py logo.emf logo.pdf

Handles MOVETOEX, POLYLINETO16, POLYBEZIERTO16, CLOSEFIGURE and FILLPATH with a solid brush or the stock white/black
brush. Anything else (text, bitmaps, clipping) is ignored and reported, so the output has to be checked by eye.
Needs matplotlib.
"""

import argparse
import struct
import sys
from collections import Counter

STOCK = {0x80000000: "white", 0x80000004: "black"}


def parse(data: bytes):
    """Yield (record type, payload) of an EMF file."""
    i = 0
    while i < len(data):
        kind, size = struct.unpack_from("<II", data, i)
        yield kind, data[i + 8:i + size]
        i += size


def paths(data: bytes) -> tuple[list[tuple[list, list, str]], tuple[int, int, int, int], Counter]:
    """([(vertices, codes, colour)], bounds, ignored record types)."""
    from matplotlib.path import Path as P
    bounds = struct.unpack_from("<4i", data, 8)
    org, ext = (0, 0), None
    objects, brush, ignored = {}, "black", Counter()
    verts, codes, out = [], [], []
    handled = {1, 14, 17, 18, 19, 21, 22, 27, 37, 39, 40, 59, 60, 61, 62, 70, 75, 88, 89, 9, 10, 11, 12, 13}
    for kind, p in parse(data):
        if kind == 10:  # SETWINDOWORGEX: logical origin of the drawing
            org = struct.unpack_from("<ii", p, 0)
        elif kind == 9:  # SETWINDOWEXTEX: logical size of the drawing
            ext = struct.unpack_from("<ii", p, 0)
        elif kind == 39:  # CREATEBRUSHINDIRECT: ihBrush, style, colour (0x00bbggrr)
            handle, style, rgb = struct.unpack_from("<III", p, 0)
            objects[handle] = "#%02x%02x%02x" % (rgb & 255, (rgb >> 8) & 255, (rgb >> 16) & 255)
        elif kind == 37:  # SELECTOBJECT
            (handle,) = struct.unpack_from("<I", p, 0)
            if handle in STOCK:
                brush = STOCK[handle]
            elif handle in objects:
                brush = objects[handle]
        elif kind == 59:
            verts, codes = [], []
        elif kind == 27:
            x, y = struct.unpack_from("<ii", p, 0)
            verts.append((x, y)), codes.append(P.MOVETO)
        elif kind in (89, 88):  # POLYLINETO16 / POLYBEZIERTO16: bounds(16) count(4) points (int16 x, y)
            (count,) = struct.unpack_from("<I", p, 16)
            pts = [struct.unpack_from("<hh", p, 20 + 4 * k) for k in range(count)]
            if kind == 89:
                verts += pts
                codes += [P.LINETO] * count
            else:
                verts += pts
                codes += [P.CURVE4] * count
        elif kind == 61 and verts:  # CLOSEFIGURE
            verts.append(verts[0]), codes.append(P.CLOSEPOLY)
        elif kind == 62 and verts:  # FILLPATH
            out.append((list(verts), list(codes), brush))
        elif kind not in handled:
            ignored[kind] += 1
    if ext:  # coordinates are logical: the drawing is the window, not the device bounds of the header
        bounds = (org[0], org[1], org[0] + ext[0], org[1] + ext[1])
    return out, bounds, ignored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emf")
    ap.add_argument("pdf")
    ap.add_argument("--png", help="also write a PNG preview")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("pdf")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as P
    shapes, _, ignored = paths(open(args.emf, "rb").read())
    points = [v for verts, _, _ in shapes for v in verts]  # crop to the drawing: the EMF canvas has a white margin
    x0, x1 = min(x for x, _ in points), max(x for x, _ in points)
    y0, y1 = min(y for _, y in points), max(y for _, y in points)
    fig = plt.figure(figsize=(3, 3 * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes((0, 0, 1, 1))
    for verts, codes, colour in shapes:
        ax.add_patch(PathPatch(P(verts, codes), facecolor=colour, edgecolor="none", linewidth=0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    fig.savefig(args.pdf, transparent=False, bbox_inches="tight", pad_inches=0)
    if args.png:
        fig.savefig(args.png, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0)
    print(f"{len(shapes)} filled paths; ignored record types: {dict(ignored) or 'none'}", file=sys.stderr)


if __name__ == "__main__":
    main()
