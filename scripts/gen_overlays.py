#!/usr/bin/env python3
"""
Generate text overlay PNGs using Pillow — no libfreetype required in FFmpeg.
Outputs to assets/: watermark.png + MOGGED letter reveals.
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1080, 1920
ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

def get_font(size, bold=False):
    candidates = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Arial.ttf',
        '/Library/Fonts/Arial.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    return ImageFont.load_default()

# ── 1. ASCENSION watermark (white, 45% opacity, bottom-centre) ───────────────
img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
font_wm = get_font(28)
text = 'ASCENSION'

# Measure text
bbox = draw.textbbox((0, 0), text, font=font_wm)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
x = (W - tw) // 2
y = H - 260

# Letter-spaced manually
spacing = 8
total_w = 0
chars = list(text)
widths = []
for ch in chars:
    b = draw.textbbox((0, 0), ch, font=font_wm)
    widths.append(b[2] - b[0])
    total_w += b[2] - b[0] + spacing
total_w -= spacing
x = (W - total_w) // 2

cur_x = x
for ch, w in zip(chars, widths):
    draw.text((cur_x, y), ch, font=font_wm, fill=(255, 255, 255, 115))
    cur_x += w + spacing

img.save(os.path.join(ASSETS, 'watermark.png'))
print(f'✅ watermark.png  (text at x={x}, y={y})')

# ── 2. MOGGED word-reveal frames ─────────────────────────────────────────────
font_big = get_font(140)
reveals = ['M', 'MO', 'MOG', 'MOGG', 'MOGGE', 'MOGGED']

for word in reveals:
    img2 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(img2)
    draw2.text((60, int(H * 0.35)), word, font=font_big, fill=(255, 255, 255, 255))
    path = os.path.join(ASSETS, f'mogged_{word}.png')
    img2.save(path)
    print(f'✅ mogged_{word}.png')

print('\nAll overlays generated.')
