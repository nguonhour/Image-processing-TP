# Image Processing TP

This project implements all five tasks with NumPy arithmetic, comparisons,
array indexing and loops. Pillow only reads/writes images and labels contact
sheets. No OpenCV arithmetic, blending, masking or filtering functions are used.

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

Tasks 1–4 default to `imagefortask1-4.png` beside the scripts. Color is converted
to grayscale manually using NumPy. Tasks 1 and 3 use a horizontally flipped
copy as B unless you supply `--b another_image.png`. Task 5 defaults to synthetic
demo frames; use the teacher's frames for submission. Results are saved under `outputs/`.

For your submission, download frames.zip from Moodle in your signed-in browser,
extract it into `frames/`, and run:

```sh
python task5.py --frames frames
```

To override the default inputs, use images of identical dimensions (8-bit grayscale, RGB or RGBA):

```sh
python task1.py --a images/A.png --b images/B.png
python task2.py --a images/A.png --n 50
python task3.py --a images/A.png --b images/B.png
python task4.py --a images/A.png
python task5.py --frames frames
```

## Results and explanations

1. **Arithmetic:** A+B, A-B, A*B and A/(B+1e-6) are computed in float64.
   Raw formula outputs are saved as `.npy` arrays, preserving negatives and
   values above 255. PNG previews saturate values to 0–255. Multiplication also
   has a `/255` preview and division a `*128` preview, where an approximately
   equal ratio appears mid-gray. These preview scalings are not the raw results.
2. **Noise averaging:** independently add Gaussian noise with mean 0 and
   standard deviation 20 to each observation, accumulate them manually and
   divide by N. Average floating-point observations before clipping. All noisy
   previews, the average, and measured mean squared errors are saved. Expected
   noise variance falls from 400 to 400/N; saved PNGs truncate fractions.
3. **Blending:** alpha 0.5 balances A and B; 0.8 favors A; 0.2 favors B.
4. **Masking:** a central rectangular ROI has mask value 1 and the rest 0.
   Multiply the image by this mask. The displayed mask uses 255 for visibility.
5. **Motion:** naturally sort frames, manually convert RGB to luminance when
   needed, subtract consecutive frames, take the absolute difference manually,
   and threshold at 25. A manually implemented 3x3 majority filter suppresses
   isolated detections. Every frame gets a difference, mask, isolated image,
   and red overlay. The first frame has no predecessor, so its mask is empty.
   This method assumes a fixed camera and relatively stable lighting. It detects
   changed areas, including previous object positions, rather than guaranteeing
   a full object silhouette. Stationary objects do not remain detected.

Each task has a `comparison.png` for quick review. Task 5 also saves motion pixel
counts and `outputs/task5/INPUTS.txt` records whether demo frames were used.

Submit `task1.py` through `task5.py`, `requirements.txt`, and the visual outputs after rerunning with
teacher frames. Keep the README with the submission to explain the calculations.

Each task file is standalone and runs only its own task. The original `lab.py` is kept as an optional all-tasks runner.
