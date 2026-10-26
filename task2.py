"""Task 2: Noise reduction by averaging. Pixel operations use manual NumPy logic."""
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





def run(a, output, n, alpha_seed=42):
    folder = output / 'task2'
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(alpha_seed)
    total = np.zeros(a.shape, dtype=np.float64)
    first = None
    for i in range(n):
        noisy = a + rng.normal(0, 20, a.shape)
        # Accumulate BEFORE clipping to avoid bias in the experiment.
        total += noisy
        if i == 0:
            first = noisy.copy()
        save(folder / f'noisy_{i:03}.png', noisy)
    average = total / n
    save(folder / 'baseline.png', a)
    save(folder / 'average.png', average)
    np.save(folder / 'average_raw.npy', average)
    def mse(image):
        error = image - a
        return float(np.sum(error * error) / a.size)
    (folder / 'metrics.txt').write_text(
        f'N = {n}\nSingle noisy MSE (unclipped): {mse(first):.4f}\n'
        f'Average MSE (unclipped): {mse(average):.4f}\n'
        f'Expected noise variance: single = 400; average = {400/n:.4f}\n')
    comparison(folder, 'comparison.png', [('Baseline', 'baseline.png'),
               ('One noisy image', 'noisy_000.png'), (f'Average of {n}', 'average.png')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--a', type=Path, default=Path(__file__).resolve().parent / 'imagefortask1-4.png', help='Baseline image; color is converted manually to grayscale')
    parser.add_argument('--n', type=int, default=50)
    args = parser.parse_args()
    if args.n < 1:
        parser.error('n must be positive')
    a = args.a
    run(load(a), args.output, args.n)
    print('Task 2 complete:', (args.output / 'task2').resolve())


if __name__ == '__main__':
    main()
