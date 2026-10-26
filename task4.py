"""Task 4: Masking and ROI isolation. Pixel operations use manual NumPy logic."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw


def load(path):
    # Decode the image with Pillow; calculate grayscale manually with NumPy.
    with Image.open(path) as image:
        if image.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError(f'{path}: use an 8-bit grayscale, RGB or RGBA image')
        data = np.array(image).astype(np.float64)
    if data.ndim == 3:
        gray = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
        if data.shape[2] == 4:
            # Composite transparent pixels onto white manually.
            alpha = data[:, :, 3] / 255.0
            gray = alpha * gray + (1 - alpha) * 255.0
        data = gray
    return data


def save(path, data):
    # Manual saturation prevents uint8 wraparound.
    data = np.where(data < 0, 0, np.where(data > 255, 255, data))
    Image.fromarray(data.astype(np.uint8)).save(path)


def sheet(path, panels):
    width, height = panels[0][1].size
    canvas = Image.new('RGB', (width * len(panels), height + 34), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(panels):
        canvas.paste(im, (i * width, 34))
        draw.text((i * width + 8, 10), label, fill='black')
    canvas.save(path)


def comparison(folder, filename, items):
    panels = []
    for label, name in items:
        with Image.open(folder / name) as im:
            panels.append((label, im.copy()))
    sheet(folder / filename, panels)





def run(a, output):
    folder = output / 'task4'
    folder.mkdir(parents=True, exist_ok=True)
    h, w = a.shape
    mask = np.zeros((h, w), dtype=np.float64)
    mask[h//4:3*h//4, w//4:3*w//4] = 1
    save(folder / 'original.png', a)
    save(folder / 'mask.png', mask * 255)
    save(folder / 'isolated.png', a * mask)
    comparison(folder, 'comparison.png', [('Original', 'original.png'),
               ('Binary ROI mask', 'mask.png'), ('Isolated region', 'isolated.png')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--a', type=Path, default=Path(__file__).resolve().parent / 'sample_inputs' / 'imagefortask1-4.png', help='Baseline image; color is converted manually to grayscale')
    args = parser.parse_args()
    a = args.a
    run(load(a), args.output)
    print('Task 4 complete:', (args.output / 'task4').resolve())


if __name__ == '__main__':
    main()
