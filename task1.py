"""Task 1: Image arithmetic. Pixel operations use manual NumPy logic."""
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





def run(a, b, output):
    if a.shape != b.shape:
        raise ValueError('A and B must have exactly the same dimensions')
    folder = output / 'task1'
    folder.mkdir(parents=True, exist_ok=True)
    eps = 1e-6
    results = {'addition': a + b, 'subtraction': a - b,
               'multiplication': a * b, 'division': a / (b + eps)}
    save(folder / 'A.png', a)
    save(folder / 'B.png', b)
    for name, result in results.items():
        np.save(folder / f'{name}_raw.npy', result)
        save(folder / f'{name}_clipped.png', result)
    # Additional readable views; these do NOT replace the raw formula results.
    save(folder / 'multiplication_scaled.png', results['multiplication'] / 255)
    save(folder / 'division_scaled.png', results['division'] * 128)
    comparison(folder, 'comparison.png', [('A', 'A.png'), ('B', 'B.png')] +
               [(name, f'{name}_clipped.png') for name in results])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--a', type=Path, default=Path(__file__).resolve().parent / 'imagefortask1-4.png')
    parser.add_argument('--b', type=Path)
    args = parser.parse_args()
    a = load(args.a)
    # A second input derived manually from A when no separate B is supplied.
    b = load(args.b) if args.b else a[:, ::-1].copy()
    run(a, b, args.output)
    (args.output / 'task1' / 'INPUTS.txt').write_text(
        f'A: {args.a}\nB: ' + (str(args.b) if args.b else 'horizontally flipped A') + '\n')
    print('Task 1 complete:', (args.output / 'task1').resolve())


if __name__ == '__main__':
    main()
