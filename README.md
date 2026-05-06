# BP Tracker Evaluator

Evaluate star tracking quality from astro photos using FWHM-based star shape metrics plus a long-trail detector.

## What It Does

The script analyzes each image and outputs:

- `tracking_score_0_100`: overall quality score (higher is better).
- `median_fwhm_px`: median star size (blur) in pixels.
- `median_elongation`: median star ellipticity ratio (1.0 is round).
- `trail_idx`: compact streakiness indicator from local star shape.
- `p95_trail`: 95th percentile trail length in pixels (captures long streaks).
- `stars`: number of star detections used.

This is designed so badly tracked exposures (long star trails) score very low, while well-tracked images score higher.

## Requirements

- Python 3.9+
- Packages:
  - `opencv-python`
  - `numpy`

## Usage

Run on a folder:

```bash
python evaluate_tracking.py path-to-images
```

Run on one or more files:

```bash
python evaluate_tracking.py images/image1 images/image2
```

JSON output:

```bash
python evaluate_tracking.py images --json
```

## Example Output

```text
Tracking quality (higher score is better):
image1.png                score= 51.90  FWHM= 6.03px  elong=1.20  trail_idx=1.00px  p95_trail=  6.0px  stars=1192
image2.png                score=  2.67  FWHM= 6.84px  elong=2.57  trail_idx=4.17px  p95_trail=102.6px  stars=1194
```

## How To Interpret Results

- **Best single metric for visible smear length:** `p95_trail`
  - Larger value means longer star trails.
- **Overall pass/fail-style metric:** `score`
  - `100` is theoretical perfect tracking.
  - In real data, excellent tracking is usually well below 100.
- **Local star sharpness metric:** `FWHM`
  - Lower is sharper.
  - Can increase from focus/seeing/lens softness even with decent tracking.
- **Directional drift metric:** `elong` and `trail_idx`
  - `elong` near `1.0` means round stars.
  - Higher values indicate elongation from motion or optical effects.

## Notes

- Scores are most comparable when camera/lens/settings are similar.
- Longer focal lengths naturally stress tracking more, so equal score across very different lenses is not expected.
- Use `score` for ranking frames, and `p95_trail` when you need a physically intuitive smear length in pixels.
