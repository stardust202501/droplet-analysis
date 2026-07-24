# Droplet Analysis System

Droplet Analysis System is a desktop and command-line program for fluorescence
microscopy images containing digital PCR droplets. It detects droplet boundaries
with Cellpose instance segmentation after green-channel image enhancement,
measures fluorescence from the original image,
separates droplets with K-means clustering, and exports per-droplet results.

## Analysis workflow

1. Read the selected fluorescence image.
2. Extract the green fluorescence channel.
3. Verify that the image contains a dominant green fluorescence signal.
4. Build a background-corrected, locally enhanced green-channel image.
5. Build a complementary image with broad positive-fluorescence halos suppressed.
6. Run the Cellpose `cyto3` model on the source image and the Cellpose `nuclei`
   model on both enhanced images.
7. Retain and merge non-duplicate Cellpose instance masks.
8. Measure an eroded inner region and a shape-matched surrounding ring on the
   original green channel.
9. Subtract the surrounding-ring mean from the inner-droplet mean to remove spatial
   illumination bias.
10. Square the positive part of the local contrast so rare bright droplets are
   not masked by small symmetric background variations.
11. Apply K-means with `k = 2` independently to each image.
12. Assign the lower-centre cluster as negative and the higher-centre cluster as
   positive.
13. Export the overlay, droplet table, and image summary.

Preprocessed images are used only for boundary detection. Fluorescence values
are measured from the original image.

## Environment

The program is validated in:

```text
C:\Users\lss\anaconda3\envs\gr_main
Python 3.9.23
```

Install the required packages in the environment with:

```bash
conda activate gr_main
pip install -r requirements.txt
```

## Start the desktop program

Double-click:

```text
run_main.bat
```

Or run:

```bash
conda activate gr_main
cd /d E:\path\to\droplet-analysis-ddpcr
python main.py
```

Select one input image, select an output folder, confirm the approximate droplet
diameter, and start the analysis. The diameter is supplied to Cellpose as an
expected object scale; output boundaries are Cellpose instance masks rather
than fixed-radius circles.

## Command-line analysis

```bash
python droplet_run.py E:\path\to\Image7.tif --output-folder E:\path\to\results
```

The default droplet diameter is `40 px`. It can be changed with:

```bash
python droplet_run.py E:\path\to\Image7.tif --output-folder E:\path\to\results --diameter 40
```

## Results

For each image, the program writes:

- `*_overlay.png`: all detected boundaries over the source image;
- `*_droplets.csv`: droplet position, radius, area, grayscale values, K-means
  cluster, and positive or negative classification.

The output folder also contains `batch_summary.csv` with image-level total,
positive, negative, edge-droplet, image-quality, and clustering results.
Images without a dominant green fluorescence channel are recorded as
`skipped_non_green` and are not clustered.

Overlay colours:

- red: positive droplet;
- cyan: negative droplet;
- yellow: droplet touching the image boundary.

Droplets touching the image boundary are outlined and recorded but excluded
from K-means clustering and positive or negative counts.

## Output columns

The per-droplet CSV includes:

- `droplet_id`: unique droplet number within the image;
- `centroid_x`, `centroid_y`: centre coordinates in pixels;
- `radius_px`: area-equivalent radius calculated from the detected contour;
- `area_px`: area enclosed by the detected contour;
- `raw_mean_gray`: mean fluorescence in the original green channel;
- `raw_median_gray`: median fluorescence in the original green channel;
- `corrected_mean_gray`: mean fluorescence after low-frequency illumination
  correction;
- `local_background_gray`: mean grayscale value in the local ring surrounding
  the droplet;
- `local_contrast_gray`: inner-droplet mean minus the local-ring mean and the
  fluorescence contrast used to construct the K-means feature;
- `kmeans_feature`: squared positive local contrast used for K-means clustering;
- `cluster`: K-means cluster number;
- `classification`: `positive`, `negative`, or `excluded_edge`.

## Verification

```bash
python -m py_compile main.py droplet_analysis.py droplet_run.py
python droplet_run.py E:\path\to\Image7.tif --output-folder E:\path\to\test_results
```
