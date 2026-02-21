#!/bin/bash
cd ~/cosmos-predict2.5
source .venv/bin/activate
python -c "
from PIL import Image
img = Image.open('assets/base/excavator1.png')
print(f'Original mode: {img.mode}, size: {img.size}')
if img.mode == 'RGBA':
    rgb = Image.new('RGB', img.size, (255, 255, 255))
    rgb.paste(img, mask=img.split()[3])
    img = rgb
elif img.mode != 'RGB':
    img = img.convert('RGB')
img.save('assets/base/excavator1.png')
print(f'Saved as RGB: {img.mode}, size: {img.size}')
"
