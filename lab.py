"""Image processing TP: manual pixel arithmetic, averaging, blending and motion.
Pillow is used only for image input/output and labeled comparison sheets.
"""
from pathlib import Path
import argparse
import re
import numpy as np
from PIL import Image, ImageDraw


def load(path):
    # Inputs must already be grayscale; no high-level grayscale conversion.
    with Image.open(path) as image:
        data = np.array(image)
    if data.ndim != 2 or data.dtype != np.uint8:
        raise ValueError(f'{path}: use an 8-bit grayscale image')
    return data.astype(np.float64)


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


def demo_inputs(root):
    """Deterministic synthetic inputs, explicitly separate from teacher data."""
    root.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((240, 320))
    a = 35 + 130 * x / 319 + 20 * y / 239
    a[(x - 100)**2 + (y - 110)**2 < 55**2] = 225
    a[155:210, 190:285] = 65
    b = 30 + 140 * y / 239
    b[35:145, 165:270] = 235
    b[(x - 75)**2 + (y - 170)**2 < 38**2] = 70
    save(root / 'A.png', a)
    save(root / 'B.png', b)
    frames = root / 'frames'
    frames.mkdir(exist_ok=True)
    rng = np.random.default_rng(12)
    background = 45 + 40 * x / 319 + 15 * y / 239
    for i in range(24):
        frame = background + rng.normal(0, 2, background.shape)
        left = 10 + i * 10
        frame[90:135, left:left + 35] = 220
        save(frames / f'frame_{i:03}.png', frame)
    return root / 'A.png', root / 'B.png', frames


def tasks_one_to_four(a, b, output, n, alpha_seed=42):
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

    folder = output / 'task3'
    folder.mkdir(parents=True, exist_ok=True)
    for alpha, name in [(0.5, 'balanced'), (0.8, 'more_A'), (0.2, 'more_B')]:
        save(folder / f'{name}.png', alpha * a + (1 - alpha) * b)
    comparison(folder, 'comparison.png', [('alpha = 0.5', 'balanced.png'),
               ('alpha = 0.8: more A', 'more_A.png'), ('alpha = 0.2: more B', 'more_B.png')])

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


def task_five(source, output, threshold):
    folder = output / 'task5'
    folder.mkdir(parents=True, exist_ok=True)
    def natural(path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(path))]
    files = sorted([p for p in source.rglob('*') if p.suffix.lower() in
                    {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}], key=natural)
    if len(files) < 2:
        raise ValueError('Task 5 needs at least two frames')
    previous = None
    first_shape = None
    rows = ['frame,moving_pixels']
    previews = []
    for i, path in enumerate(files):
        with Image.open(path) as im:
            raw = np.array(im)
        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            rgb = raw[:, :, :3].astype(np.float64)
            current = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        elif raw.ndim == 2:
            current = raw.astype(np.float64)
        else:
            raise ValueError(f'Unsupported frame format: {path}')
        if first_shape is None:
            first_shape = current.shape
        if current.shape != first_shape:
            raise ValueError('All frames must have equal dimensions')
        difference = np.zeros_like(current) if previous is None else current - previous
        difference = np.where(difference < 0, -difference, difference)
        mask = (difference > threshold).astype(np.uint8)
        # Manual 3x3 majority filter: suppress isolated noisy detections.
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:h+1, 1:w+1] = mask
        votes = np.zeros((h, w), dtype=np.uint16)
        for dy in range(3):
            for dx in range(3):
                votes += padded[dy:dy+h, dx:dx+w]
        mask = (votes >= 5).astype(np.uint8)
        save(folder / f'{i:04}_difference.png', difference)
        save(folder / f'{i:04}_mask.png', mask * 255)
        save(folder / f'{i:04}_isolated.png', current * mask)
        overlay = np.stack([current, current, current], axis=2)
        overlay[mask == 1] = [255, 40, 40]
        save(folder / f'{i:04}_overlay.png', overlay)
        rows.append(f'{path.name},{int(np.sum(mask))}')
        if i in {0, len(files)//2, len(files)-1}:
            with Image.open(folder / f'{i:04}_overlay.png') as im:
                previews.append((f'Frame {i}: motion in red', im.copy()))
        previous = current
    sheet(folder / 'comparison.png', previews)
    (folder / 'motion_counts.csv').write_text('\n'.join(rows) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path)
    parser.add_argument('--b', type=Path)
    parser.add_argument('--frames', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=25)
    args = parser.parse_args()
    if args.n < 1 or args.threshold < 0:
        parser.error('n must be positive and threshold must be nonnegative')
    if bool(args.a) != bool(args.b):
        parser.error('Supply both --a and --b, or neither')
    if args.a is None or args.frames is None:
        demo_a, demo_b, demo_frames = demo_inputs(Path('sample_inputs'))
    a_path, b_path = (args.a, args.b) if args.a else (demo_a, demo_b)
    frames = args.frames if args.frames else demo_frames
    args.output.mkdir(parents=True, exist_ok=True)
    tasks_one_to_four(load(a_path), load(b_path), args.output, args.n)
    task_five(frames, args.output, args.threshold)
    (args.output / 'INPUTS.txt').write_text(
        f'A: {a_path}\nB: {b_path}\nFrames: {frames}\n'
        f'Task 5 uses {"teacher/user-provided" if args.frames else "SYNTHETIC DEMO"} frames.\n')
    print(f'Completed five tasks. Results: {args.output.resolve()}')
    print((args.output / 'task2' / 'metrics.txt').read_text())
    if not args.frames:
        print('Task 5 is a DEMO. Run again with --frames pointing to teacher frames.')


if __name__ == '__main__':
    main()
