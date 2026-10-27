# SPIOMetrologyViewer

Finds dirt and defects in microscope images of the devices on a wafer, using classical image
processing plus hand-set rules. No training data is needed. Every detection is saved as a crop
with features, so a labelled dataset builds up as you use it.

## Getting started

The code is on GitHub; the data is in a shared **workspace** folder on Nextcloud: `measurements.yaml`,
`configs/`, `tolerances/`, `tunings/`, the position maps, `Images/` and `output/`. Everyone works
in the same workspace, so tunings, tolerances and results are shared as soon as Nextcloud has
synced them.

One-time setup on a Windows PC (Linux: see below):

1. Install [Git for Windows](https://git-scm.com) and [Python 3.14](https://www.python.org/downloads/)
   (keep "py launcher" ticked).
2. In the Nextcloud desktop client, sync the shared `24 SPIOMetrologyViewer` folder.
3. Clone the code to a short path, e.g. in a command prompt:
   `cd %USERPROFILE%\Documents` then `git clone https://github.com/SPIO-Victor/SPIOMetrologyViewer.git`.
4. Double-click `start.bat` in the clone. The first start installs the Python packages (about
   2 minutes). Then it asks for the workspace: choose the Nextcloud `24 SPIOMetrologyViewer` folder. The
   choice is saved in `settings.yaml` in the clone. Delete that file to choose again.

After that, start the program with `start.bat` (make a desktop shortcut to it). Every start fetches
the latest version from GitHub (`git pull`) and installs new packages if `requirements.txt` changed.

On **Linux Mint** (or Ubuntu), in a terminal:

1. Python 3.14 with tkinter, from the deadsnakes PPA:
   ```
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt install git python3.14 python3.14-venv python3.14-tk
   ```
   Without sudo, [uv](https://docs.astral.sh/uv) also works (`curl -LsSf https://astral.sh/uv/install.sh | sh`),
   but its Python's Tk can only use old bitmap fonts, so the GUI looks rough. `start.sh` prefers the
   system Python 3.14 and switches an existing uv `.venv` over to it.
2. Get the Nextcloud files to `~/Nextcloud`, either way:
   - Sync the `24 SPIOMetrologyViewer` folder with the Nextcloud desktop client (`sudo apt install nextcloud-desktop`).
   - Or, without a full local copy, mount Nextcloud with rclone (only opened files are cached, at most 50 GB).
     In Nextcloud, Settings → Security → "Create new app password", then:
     ```
     sudo apt install rclone
     rclone config create nextcloud webdav vendor=nextcloud url=https://nextcloud.spiosystems.com/remote.php/dav/files/<your login>/ user=<your login> pass=<app password> --obscure
     mkdir -p ~/.config/systemd/user && cp nextcloud-rclone.service ~/.config/systemd/user/
     systemctl --user enable --now nextcloud-rclone
     ```
     (run the `cp` in the clone, after step 3). It then mounts at every login.

   The Nextcloud entry of the desktop's "Online Accounts" doesn't work: the program can't create
   files or folders through it, so batch runs and saving new tunings or tolerances fail.
3. `cd ~/Documents && git clone https://github.com/SPIO-Victor/SPIOMetrologyViewer.git`
4. `cd SPIOMetrologyViewer && ./start.sh`. The first start takes a few minutes. Choose the
   Nextcloud folder when asked, e.g. `~/Nextcloud/SPIO Systems/8 Software/24 SPIOMetrologyViewer`.

`start.sh` does the same as `start.bat`. For a menu or desktop shortcut, edit the paths in
`spiometrologyviewer.desktop` and copy it to `~/.local/share/applications/`. Run
`./start.sh --console` to keep the program in the terminal and see its error messages. Without uv,
`start.sh` uses the system Python; that needs `sudo apt install python3-venv python3-tk`.

Working together:

- **Batch runs:** a running batch leaves `RUNNING.txt` (who, when) in the type's output folder.
  "Run on all images" warns if someone else is running the same type. Running it twice at once
  overwrites results and makes Nextcloud conflict copies.
- **Tunings** are named after who made them (`..._20261026_101500_vit.yaml`).
- **Saved parameters:** "Save parameters..." over a file in `configs/` changes the defaults for
  everyone.
- **Image paths** in `results.csv` are relative to the workspace, so results open on every PC.

### Releasing changes (for the maintainer)

Commit and push to `main` (PyCharm: Git → Commit / Push). Colleagues get the change the next time
they start the program. When a change can alter the results, bump `__version__` in
`spiometrologyviewer/__init__.py`. It goes into `results.csv` as `code_version`, and Statistics warns when
the ticked wafers were made with different versions (`?` = before versions were recorded).

When adding a package, add it with its version to `requirements.txt`. `workspace_template/` holds
the defaults for a new, empty workspace. The live configs and tolerances are those in the workspace.

Where the workspace is found: the environment variable `SPIOMETROLOGYVIEWER_WORKSPACE`, else
`settings.yaml` (`workspace: <folder>`), else the repo folder itself. Data in the repo folder is
ignored by git (see `.gitignore`).

## Plates and measurement types

The wafers go through several plates (P2, P3, ...), and each plate has images of one or more
**measurement types**. Each type has its own images, parameters, tolerances and results. They are
listed in `measurements.yaml`:

| Plate | Type | Images | Analysed |
|---|---|---|---|
| P2 | With mirror | `Images/P2/WithMirror/<wafer>/` | the mirror rectangle, plus its offset to the lens |
| P2 | Without mirror | `Images/P2/WithoutMirror/<wafer>/` | the lens surface (no mirror, so no offset, rotation or edge chips) |
| P3 | Alignment | `Images/P3/<wafer>/Alignment/` | offset of the dark-line pattern vs the bright-line pattern; particles, fibres, rings, stains |
| P3 | Grating | `Images/P3/<wafer>/Grading/` | particles, streaks (comets) and discoloured spots on the grating |
| P3 | Mirror | `Images/P3/<wafer>/Mirror/` | small particles and large defects on the cream mirror (gold strip ignored) |
| P4 | 3D profile | the project's metrology folder, `…/05 Metrology/17013_2/P4/<wafer>/` (`.lext`) | lens shape from LEXT 3D scans: radius of curvature, residual to the sphere, Zernike terms, stitching, bumps and pits |

Per type:

- `configs/<type>.yaml`: pipeline parameters. `analysis:` sets which analysis runs (`mirror`,
  `lens`, `alignment`, `grating`, `p3_mirror`; see `spiometrologyviewer/analyses.py`). The P2 configs
  have no `analysis` key and use `region: "mirror"` / `"lens"` instead. Each config file is also
  the template it is saved from (it keeps the comments).
- `tolerances/<type>.yaml`: pass / warn / fail cutoffs.
- `output/<plate>/<type>/<wafer>/`: results.

Only `*COLOR2D` images are analysed for P2 and P3 (`input.image_glob`: `.tif` for P2, `.jpg` for P3).
Wafers that only have LSM2D laser scans (`17013_2_P2_102testBeforeCleaning`,
`17013_2_T2_102_afterCleaning`) are left out for now. P4 reads the LEXT 3D scans (`*_P4_*.lext`, one
per lens) where the microscope saved them, in the project folder: `images:` in `measurements.yaml` is
relative to the workspace (`../../1 Projects/…`), so it works on every PC that has the `SPIO Systems`
folder in its Nextcloud. Each scan is ~120 MB (4 GB per wafer), read over the network.

### Adding a plate or measurement type

Add it to `measurements.yaml`, copy a config of the same analysis to `configs/`, and restart the GUI.
The tolerances file is created when you first press "Save" in its wafer map. If the type folder
sits inside the wafer folder (`Images/P3/<wafer>/Mirror`), write `images: Images/P3/*/Mirror`;
the wafer is then named after the `*` folder.

## Run

`start.bat` starts the GUI. From a command prompt in the clone (after `start.bat` has set up `.venv`):

```
.venv\Scripts\python -m spiometrologyviewer.gui                                   # the GUI
.venv\Scripts\python -m spiometrologyviewer.cli --measurement "P2/With mirror"    # batch for one type
.venv\Scripts\python -m spiometrologyviewer.cli --all --skip-done                 # every type, new wafers only
.venv\Scripts\python tools\cluster_crops.py output\P2\WithMirror\<wafer> --k 8   # optional: group crops for review
```

On Linux, use `.venv/bin/python` and `/` in the paths.

A wafer is every folder in the type's image folder (at any depth) that directly holds at least
`input.min_images` (10) matching images; it is named after its folder. `--skip-done` skips wafers
that already have a `results.csv`. A wafer of 33 images takes about 20 s. The old form,
`cli <image folder> --out <folder> [--config ...]`, still works.

## GUI

One tab per plate; inside it, one tab per measurement type; inside that:

- **Statistics** (opens first): plots comparing the type's wafers (below).
- **Wafer map**: one wafer at a time, device by device.
- **Step-by-step tuning**: the pipeline on one image, with that type's parameters. It is only
  loaded when you first open it.

## Step-by-step tuning

Goes through the pipeline one step at a time on one image: the image (with that step's result)
on the left, the step's parameters on the right. Changing a parameter recomputes and redraws
straight away; **Next step** moves on. "Show" switches between views of the step (e.g. the
flattened image, or the texture energy behind the residue threshold). Use `<` / `>` at the top
to check the same parameters on other images. Mouse wheel zooms, drag pans, double-click fits.
For "Without mirror", step 2 is **Find the lens** instead of the mirror and lens steps.

Defaults come from the type's config file; changed values are marked with their default in
orange. On the last step:

- **Save parameters...** writes them to a YAML file. Overwrite the type's file in `configs/` to
  make them its new defaults.
- **Run on all images...** runs the batch with them over all the type's wafers into its output
  folder. If some wafers already have results, it asks whether to run those again. When it
  finishes, Statistics and Wafer map reload.

Every batch run also saves the parameters it used as `config_used.yaml` next to the results.

## Tunings

Every set of parameters that is run or saved is kept as a **tuning**: a timestamped copy in
`tunings/<config name>/`, e.g. `tunings/P2_with_mirror/P2_with_mirror_20261026_101500_vit.yaml`
(time and user who made it; older ones have no user).

- Running or saving parameters identical to an existing tuning reuses its name. Anything else
  creates a new file. The `output` section (crop and thumbnail sizes) doesn't change the numbers,
  so it isn't part of the comparison.
- The batch writes the tuning's name into every row of `results.csv` (column `tuning`). Older
  results without that column are named by matching their `config_used.yaml` against the tunings.
- The tuning tab shows, at the top, which tuning the current parameters are. While you change
  them it shows "new" until you save or run them.
- **Wafer map** shows the tuning of the wafer on screen.
- **Statistics** shows:
  - the tuning under each wafer and in the summary table
  - at the bottom of the plot, which also appears in saved images, the tuning of what is plotted
  - a red warning when the ticked wafers were analysed with different tunings
- "Run on all images" also warns before you run only the new wafers when the existing ones were
  made with another tuning.

## Wafer map

**Wafer map** shows every device of one wafer at its position from
`DevicePositionMap.xlsx`, as a thumbnail of its image in a coloured frame. The device number is
the last number in the image name: `..._P2_001_COLOR2D` is device 1. Y points down by default;
tick "Flip Y" to flip it.

- **Data**:
  - The wafers are the subfolders with a `results.csv` in the type's output folder. **Wafer**
    (or `<` / `>`) picks the one shown. "Results folder..." picks another batch output folder.
  - Statistics and counts are for the wafer shown; the tolerances apply to every wafer of the type.
- **Thumbnail**: "Raw" (the image), "Result" (with detections outlined) or "Off" (plain coloured
  blocks). The bottom strip shows the device number and the failures / the chosen metric's value.
- **Colour**: each device's frame is green (pass), yellow (warn) or red (fail). The rule per metric is
  `value <= Pass` → pass, `value > Fail` → fail, anything in between → warn. Offsets and
  rotation are compared as absolute values.
- **Colour by**: "Overall" (the default) uses each device's worst status over the ticked
  metrics. You can also pick a single metric.
- **Tolerances**: one row per metric:
  - the dataset's min, mean, max and median
  - how many devices pass, warn, fail or have no data
  - the Pass and Fail cutoffs

  Changing a cutoff recolours the map (and the Statistics tab) straight away. "Save" writes the
  type's `tolerances/<type>.yaml`, which is loaded at start-up; "Reset to defaults" restores the
  built-in defaults. "Without mirror" has no offset, rotation or edge chip metrics.
- **A device**:
  - Click a device to see its values. Double-click it (or use "Open device...") to open a
    window with the result image or the raw image, all its results with their status, and
    every detected object.
  - Click an object to mark it on the image, or double-click it to zoom in. Column headers
    sort the list.
  - "Open in step-by-step tuning" loads the device's image into the first tab.

## Wafer statistics

The **Statistics** tab compares the wafers of one measurement type. On the left:
tick the wafers to include (each has a fixed colour), choose a view and the parameter. The
parameters are the tolerance metrics, every other numeric `results.csv` column, and the offsets
and rotation with their sign (`offset_x_um (signed)`). Yes/no columns such as `particles_present`
count as 1 / 0, so their mean is the share of devices.

| View | Shows |
|---|---|
| Yield bars | % (and number) of devices that pass / warn / fail per wafer, for the overall status or one metric |
| Distribution curves | Per wafer, the % of devices at or below each value; the dashed lines are the pass / fail cutoffs |
| Average wafer map | Per device position, the mean / median / max / std over the ticked wafers, or the % of wafers where it fails. Hover a square for details |
| Parameter vs parameter | One dot per device, coloured by wafer, with the Spearman rank correlation per wafer. Click a dot to see which device it is; double-click opens it |

Below the plot is a summary table per wafer (n, mean, median, std, min, max, % pass / warn / fail),
which "Export table CSV..." saves. The toolbar zooms, pans and saves the plot as an image.
Changing a tolerance in the Wafer map tab updates the plots straight away. Double-clicking a dot in
"Parameter vs parameter" shows that wafer in the Wafer map tab and opens the device.

## Output

| File | Content |
|---|---|
| `output/<plate>/<type>/wafers.csv` | The wafers of the last run, with their image folders. The files below are in `output/<plate>/<type>/<wafer>/` (shortened to `output/` here) |
| `output/results.csv` | One row per image: `dirt_coverage_inner_pct`, `dirt_coverage_edge_pct`, counts and `*_present` flags for particles / fibres / residue / edge chips, `saturated_pct_inner`, and the alignment columns below. Without a mirror: `lens_area_mm2` and only the lens columns, no edge chips |
| `output/objects.csv` | One row per detected object: class from the rules, features, and an empty `label` column for you to fill in |
| `output/overlays/*.png` | Each image with detections outlined: red = particle, blue = fibre, magenta = residue, orange = edge chip, cyan = unknown. Bright green = analysed area, yellow = inner/edge boundary, hatched = overexposed |
| `output/crops/*.png` | One crop per object, named `<object_id>.png` |
| `output/thumbs/*.jpg`, `output/thumbs_result/*.jpg` | Small raw / result images for the wafer map (`output.thumb_px`) |
| `output/plots/*.png` | P4: per lens, R vs aperture, the radial residual profile, the Zernike terms and the mean stitch tile ("Plots" in the device window) |
| `output/clusters/*.png` | Crop grids per feature cluster (from `cluster_crops.py`) |
| `output/config_used.yaml` | The parameters this run used (its tuning is the `tuning` column of `results.csv`) |

### Alignment columns (`results.csv`)

| Column | Meaning |
|---|---|
| `offset_x_um`, `offset_y_um` | Mirror centre minus lens centre, in image axes: **+x = mirror right of the lens centre, +y = mirror below it** |
| `offset_um` | Length of that offset |
| `mirror_cx_px`, `mirror_cy_px`, `lens_cx_px`, `lens_cy_px` | The two centres in image pixels |
| `lens_diameter_um`, `lens_found` | Fitted lens (outer edge of the dark round lens) |
| `mirror_angle_deg` | Mirror rotation against the image axes |
| `lens_ring_offset_um` | Quality check: centre of the thin bright ring inside the lens, fitted separately, vs the lens centre. Normally 0–3 um; much more means one of the fits went wrong |

The mirror centre is the middle of the fitted rectangle. For images where a mirror side was placed
from the expected size (`mirror_sides_from_size`), the centre along that axis relies on
`mirror.size_um`, so it is less certain.

## How it works

1. Mask the burned-in green/yellow grid and scale bar, then fill those pixels in.
2. Fit the mirror rectangle from log-brightness edges. The mirror is the innermost edge going
   out from the centre. If an edge is lost in shadow, it is placed using the known mirror size
   (`mirror.size_um`; reported in `mirror_sides_from_size`).
3. Fit the round lens: on 720 rays from the mirror centre, the strongest dark-to-bright step is
   its outer edge; a circle is fitted while dropping outliers (reflections, dirt, cables).
   The mirror centre minus the lens centre gives the alignment offset.
4. Leave out of the analysis: the bevel rim (`border_ignore_um`), the chamfered corners
   (`corner_exclude_um`), and reflection arcs from the housing plus their shadows.
5. Even out the lighting (`ratio = image / background`). Dark objects are `ratio < dark.t_ratio`.
6. **Residue**: fine relief texture (band-pass energy) and dense clusters of small dark blobs.
   **Edge chips**: dark notches reaching `edge_chip.min_depth_um` into the mirror outline.
   **Fibre vs particle**: by elongation and skeleton shape.

All thresholds are in the type's config file.

### P3

Every step below has its own step in the tuning tab, with its parameters and views.

**Alignment** (5 marks per wafer):
1. **Line patterns**: thin bright lines (top-hat filter) and thin dark lines (black-hat filter).
   Only long straight vertical and horizontal pieces are kept, so dirt, rings and the white
   square drop out. Dark lines are bridged where a bright line crosses them. The line positions
   come from the column and row profiles.
2. **Offset**: each pattern is mirror-symmetric about its centre. The centre is the position that
   most of its lines have a mirror partner about, so missing or extra lines hardly matter.
   - `offset_x_um`, `offset_y_um` = dark centre − bright centre (+x right, +y down in the image).
   - `fit_quality` = the smallest share of lines that found a partner. `offset_ok` = no when it
     is below `lines.min_quality`.
3. **Dirt**: the lines are filled in from their surroundings, so dirt crossing a line stays one
   piece. The background is the lower of a median along x and one along y, which keeps the shaded
   channels between close lines out of the dirt. Dark objects are then classed as:
   - **ring**: a droplet or bubble rim, whose outline encloses much more than itself
   - **fibre**
   - **particle**
   - **stain**: a broad, faint darkening

   Dirt is reported inside the mark (`dirt_coverage_mark_pct`) and outside it.

**Grating**:
- **Region**: the dark olive area. The light background and the overexposed white rectangle
  drop out.
- **Dark objects**: cores below `dark.t_ratio`, grown into touching fainter pixels, so a comet
  keeps its tail. Long thin ones are **streaks**, the rest **particles**. Objects touching the
  edge of the grating are dropped (the shading along it).
- **Discoloured spots**: colour difference from the grating's median colour in hue and saturation
  only (Lab a/b, so shading doesn't count), smoothed. Clean grating stays below about 5; the
  green and blue spots reach 15–30.

**Mirror**:
- **Region**: the cream square, found by colour (blue at least `mirror.min_blue_ratio` of red).
  Its sides are where the share of mirror-coloured pixels per column or row drops, so the gold
  strip and pale patches on it are left out.
- **Dark objects**: as for the grating. Objects at least `large_min_diam_um` wide or
  `large_min_length_um` long (largest extent) are **large defects** (fibres, flakes, debris);
  the rest are **particles**.
- **Mirror width**: `mirror_width_um` varies along the wafer (about 1610–1740 µm, height 2080 µm),
  because the strip covers more or less of the mirror.

The grid overlay on P3 Grating and Mirror is found by its colour relative to the surroundings
(`overlay.local_px`), because the mirror itself is yellowish.

### P4 3D profile

One LEXT `.lext` scan per lens (Evident OLS5500, 20x objective, 9 x 9 stitched tiles; a TIFF with
the height map, the laser intensity and an invalid-pixel mask). Lengths are µm, residual heights nm.

1. **Load**: the height map in the file's own scale (`HeightDataPerPixelX/Y/Z`), times our
   **calibration factors**: from a LEXT CSV export in the wafer folder (its header has
   `X Calibration  = ,100.697519,` etc.; only the header is read) if there is one, else
   `calibration:` in the config (now X 100.697519 %, Y 100.516137 %, Z 100 %). R scales with the
   square of the lateral factor, so these matter: +0.6 % laterally is +1.2 % in R. The factors and
   their source go into `results.csv` (`cal_x_pct`, `cal_y_pct`, `cal_z_pct`, `cal_source`).
   The map is binned (`bin_px` 4, ~3.9 µm), and pixels far off their neighbours (the steep rim)
   are left out.
2. **Level**: a plane fitted to the corners outside the lens (beyond `level.corner_radius_um` from
   the scan centre), with outliers clipped, is subtracted (`tilt_x_pct`, `tilt_y_pct`, `corner_rms_um`).
3. **Lens**: the rim is the lowest point along 360 rays from the apex, fitted with a circle
   (`lens_diameter_um`, `rim_roundness_um`, ±~10 µm), `sag_um` = apex − rim. The analysed aperture
   ends `lens.edge_margin_um` inside the rim (`aperture_um`).
4. **Sphere**: geometric least-squares sphere fit (shortest distance of each point to the sphere,
   robust to outliers) to `sphere.fit_frac` (0.9) of the aperture radius: `roc_um`, and
   `roc_dev_pct` vs `nominal_roc_um` (16198). **R vs aperture**: the centre is kept and R is refitted
   to 10 %, 20 %, … 100 % of the aperture radius (`roc_10pct` … `roc_100pct`); a flat curve means a
   spherical lens, `roc_spread_pct` = max − min from 30 % on. A conic fit gives `conic_roc_um` and
   `conic_k` (0 = sphere).
5. **Shape**: the residual to the sphere (normal to it; `residual_rms_nm`, `residual_pv_nm`) is split
   into Zernike terms over the aperture (Noll, up to `zernike.n_max` 6): `astig_nm` (saddle / warp),
   `coma_nm`, `trefoil_nm`, `spherical_nm` (rms values), all of them together `lowfreq_rms_nm`, and
   the rest `midfreq_rms_nm` (stitching, defects, noise).
6. **Stitching**: the tile period is the strongest repeat of the residual's structure (2D
   autocorrelation), searched around the tile step in the scan's `matl.omp2info`. Folding the residual
   with it averages all tiles into one mean tile; its rms is `stitch_rms_nm`, the error every tile
   shares. Each tile also has its own offset and tilt, which this does not capture.
7. **Bumps and pits**: local deviations from the residual (grey-level top-hat over
   `defect.window_um`) higher / deeper than `defect.min_height_nm` (`bump_count`, `pit_count`,
   `bump_max_height_nm`, `pit_max_depth_nm`, `defect_max_nm`). They are in `objects.csv` with
   `height_nm`.

The result image is the residual map (red above the sphere, blue below; scale in the text at the
bottom left) with bumps outlined black and pits white.

## Known limits

- In the overexposed centre (hatched, 20–80% of the inner area) faint residue cannot be seen,
  so residue numbers are a lower bound there. Dark particles are still found.
- Smallest reliable object ≈ 5 µm (2 px at 2.51 µm/px).
- Residue area is the *affected area* (a closed outline), not the exact material pixels.
- Droplet fields with widely spaced droplets can come out as several "particles".
- Dirt touching a corner triangle or a reflection arc is not counted.
- With mirror: tuned on wafer `17013_2_P2_08`. On wafers where the mirror sits far off-centre,
  its edge can be hidden behind the dark lens rim and the fit can go wrong (check devices with a
  large offset in their result image).
- Without mirror: only the analysed area is set up (the disk inside the lens's bright ring, minus
  `lens.inner_margin_um` = 60 um, which keeps the dark band along the ring out). The detection
  thresholds are still the mirror's; faint dark specks on the lens are missed at `dark.t_ratio`
  0.6, so this type still needs a tuning pass.
- P3: first defaults, set by eye on contact sheets of all three wafers, not against labelled data.
  - **Alignment**: the offset was checked against the line positions on all 15 marks. Rings cut
    in two by a line can come out as fibres. Very faint stains (such as the blurred brown spot on
    19 / mark 1) are just at the threshold.
  - **Grating**: faint green hazes below the colour-difference threshold are not counted as
    discoloured.
  - **Alignment wafer map**: shows the marks in a row until `AlignmentMarkPositions.xlsx`
    (columns Position, X, Y, like `DevicePositionMap.xlsx`) is added.
- P4 3D profile: first defaults, checked on a synthetic lens (R, tilt, astigmatism, a bump and a
  stitch ripple are recovered to < 0.01 % / a few %) and on wafer `17013_2_P4_26`.
  - The apex of every scan shows concentric rings (a measurement effect of the laser microscope on
    the smooth apex) and the centre sits ~200 nm below the sphere; R fitted to small apertures
    (10–20 %) is therefore unreliable. `lens.apex_exclude_um` leaves a disk around the apex out.
  - The stitching measure is the pattern all tiles share; per-tile offsets and tilts are not
    modelled, so `stitching.remove` (subtract the mean tile before looking for defects) is off.
  - A regular grid of small marks (about every 530 µm) is counted as pits; whether they are on the
    lens or come from the microscope is not known yet.
  - Peak memory ~1.3 GB per scan (the full-resolution file is read before binning).

## Building labels → a trained model

1. Fill in `label` in `objects.csv` (look at `crops/` or the `clusters/` grids). A good
   starting point is to correct the rule class, and add new names for things that keep recurring
   (e.g. `droplet`, `rod`).
2. With about 30–50 labels per class, train a random forest on the same feature columns and
   swap it in for `classify.classify_dark`.
3. Much later, a CNN or segmentation model can be trained, using these masks as starting
   annotations.
