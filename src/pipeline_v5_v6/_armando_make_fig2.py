"""Move the 'consensus n=31' annotation in Figure 2 clear of the threshold line.

No plotted value is altered. The annotation glyphs are lifted out of the image,
the background they covered is rebuilt (grid lines, the fold-2 curve segment,
and the dashes of the threshold line, the latter copied verbatim from a clean
stretch of the same line), and the same two-line label is redrawn right-aligned
in the clear space to the left of the dashed line.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
SRC = str(_REPO / "figures_v5" / "v5_rfe_curve_pre_armando.png")
OUT = str(_REPO / "figures_v5" / "v5_rfe_curve.png")

ACCENT = np.array([231, 111, 81])      # threshold line + annotation colour
GRID = np.array([231, 231, 231])
FOLD2 = np.array([255, 152, 62])       # 'outer fold 2' curve

X0, X1, Y0, Y1 = 1600, 2000, 1100, 1240      # area holding the old annotation
DASH_SRC = (1720, 1766, 1242, 1274)          # a clean dash: x0, x1, y0, y1
DASH_RUNS = [1103, 1149, 1196]               # top row of each dash in the area
CURVE_BAND = (1868, 1906)                    # columns owned by the fold-2 curve
GRID_ROWS = (1189, 1194)
GRID_COLS = [(1632, 1637), (1869, 1874)]
RIGHT = 1712                                 # right edge of the moved label

src = Image.open(SRC)
dpi = src.info.get("dpi", (450, 450))
a = np.array(src.convert("RGB")).astype(np.int16)

sx0, sx1, sy0, sy1 = DASH_SRC
dash_block = a[sy0:sy1, sx0:sx1].copy()

# --- 1. erase everything accent-toned in the annotation area ------------------
reg = a[Y0:Y1, X0:X1].astype(float)
alpha = (255.0 - reg) / (255.0 - ACCENT)          # white-to-accent blend factor
amean = alpha.mean(axis=2, keepdims=True)
mask = (amean[..., 0] > 0.05) & (np.abs(alpha - amean).max(axis=2) < 0.16)
warm = (reg[..., 0] - reg[..., 2] > 10) & (reg[..., 0] > 190)
warm[:, CURVE_BAND[0] - X0:CURVE_BAND[1] - X0] = False   # that band is orange
mask |= warm

reg_out = a[Y0:Y1, X0:X1]
reg_out[mask] = 255

# the last 's' of 'consensus' sat on top of the fold-2 curve; wipe that block
# back to white and rebuild both the grid line and the curve inside it.
CB0, CB1 = CURVE_BAND
CY0, CY1 = 1112, 1170
reg_out[CY0 - Y0:CY1 - Y0, CB0 - X0:CB1 - X0] = 255

# --- 2. rebuild the grid lines that ran under the label -----------------------
paintable = np.ones(reg_out.shape[:2], dtype=bool)
paintable[:, CB0 - X0:CB1 - X0] = False
paintable[CY0 - Y0:CY1 - Y0, CB0 - X0:CB1 - X0] = True
for y in range(*GRID_ROWS):
    reg_out[y - Y0][paintable[y - Y0]] = GRID
for c0, c1 in GRID_COLS:
    for x in range(c0, c1):
        if X0 <= x < X1:
            reg_out[:, x - X0][paintable[:, x - X0]] = GRID
a[Y0:Y1, X0:X1] = reg_out

# --- 3. rebuild the fold-2 curve segment the word 'consensus' had covered -----
# straight, near-vertical run; the centre line is read from untouched rows
# above and below (x = 1881 at y = 1120, x = 1885 at y = 1160)
SS, PX, PY, PN = 8, 1860, 1104, 80
m = Image.new("L", (PN * SS, PN * SS), 0)
ImageDraw.Draw(m).line(
    [((1879.8 - PX) * SS, (1108 - PY) * SS), ((1886.4 - PX) * SS, (1174 - PY) * SS)],
    fill=255, width=7 * SS,
)
cov = (np.array(m.resize((PN, PN), Image.BOX)).astype(float) / 255.0)[..., None]
patch = a[PY:PY + PN, PX:PX + PN]
a[PY:PY + PN, PX:PX + PN] = np.round(patch * (1 - cov) + FOLD2 * cov)

# --- 4. restore the threshold-line dashes from a clean stretch of the line ----
for top in DASH_RUNS:
    a[top:top + (sy1 - sy0), sx0:sx1] = dash_block

img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
draw = ImageDraw.Draw(img)

# --- 5. redraw the label, right-aligned clear of the dashed line --------------
font_path = os.path.join(os.path.dirname(matplotlib.__file__),
                         "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
size = 1
while draw.textlength("consensus", font=ImageFont.truetype(font_path, size + 1)) <= 323:
    size += 1
font = ImageFont.truetype(font_path, size)

for text, top in (("consensus", 1124), ("n=31", 1177)):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((RIGHT - bbox[2], top - bbox[1]), text, font=font, fill=tuple(ACCENT))

img.save(OUT, dpi=dpi)
print("wrote", OUT, img.size, dpi, "| glyph size", size)
