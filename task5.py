"""Task 5: Moving object detection. Pixel operations use manual NumPy logic."""
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
    """Generate synthetic demonstration frames, not teacher frames."""
    root.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((240, 320))
    frames = root / 'frames'
    frames.mkdir(exist_ok=True)
    rng = np.random.default_rng(12)
    background = 45 + 40 * x / 319 + 15 * y / 239
    for i in range(24):
        frame = background + rng.normal(0, 2, background.shape)
        left = 10 + i * 10
        frame[90:135, left:left + 35] = 220
        save(frames / f'frame_{i:03}.png', frame)
    return frames


def run(source, output, threshold):
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
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--frames', type=Path)
    parser.add_argument('--threshold', type=float, default=25)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error('threshold must be finite and nonnegative')
    frames = args.frames if args.frames else demo_inputs(Path('sample_inputs'))
    run(frames, args.output, args.threshold)
    (args.output / 'task5' / 'INPUTS.txt').write_text(
        f'Frames: {frames}\n' + ('User-provided frames\n' if args.frames else 'SYNTHETIC DEMO frames\n'))
    if not args.frames:
        print('DEMO frames used. For submission, run with --frames frames.')
    print('Task 5 complete:', (args.output / 'task5').resolve())


if __name__ == '__main__':
    main()
