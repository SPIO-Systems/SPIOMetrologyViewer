"""Wafer map tab: every device of the chosen wafer in its position, shown as a thumbnail with a
pass / warn / fail coloured border, with dataset statistics, and a window with all data for one
device."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from . import io, report, tuning, wafer, workspace
from .gui import PANEL_WIDTH, ImageView, ScrollFrame, loading, ready, status

COLORS = {"pass": "#43a047", "warn": "#fbc02d", "fail": "#e53935", "nodata": "#bdbdbd"}
LIGHT = {"pass": "#dcf2dc", "warn": "#fff3c4", "fail": "#fadbd8", "nodata": "#eeeeee"}
NAMES = {"pass": "pass", "warn": "warn", "fail": "fail", "nodata": "no data"}
TEXT_ON = {"pass": "white", "warn": "black", "fail": "white", "nodata": "black"}  # readable on COLORS
OVERALL = wafer.OVERALL
THUMBS = {"Raw": "thumbs", "Result": "thumbs_result", "Off": None}   # thumbnail choice -> batch subfolder
OBJECT_COLUMNS = [("class", "Class", 70), ("label", "Label", 60), ("area_um2", "Area um2", 70),
                  ("eq_diam_um", "Diam um", 60), ("length_um", "Length um", 65), ("chip_depth_um", "Chip um", 55),
                  ("edge_dist_um", "Edge dist", 60), ("cx", "x px", 50), ("cy", "y px", 50)]


def fmt(v):
    """Short display text for a value from a results / objects table."""
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "-" if not np.isfinite(v) else f"{v:.4g}"
    return str(v)


def _isnan(v):
    return isinstance(v, (float, np.floating)) and not np.isfinite(v)


def _bgr_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _pitch(values):
    d = np.diff(np.unique(np.round(values, 6)))
    return float(d[d > 0].min()) if (d > 0).any() else 1.0


def _make_thumb(out_dir, row, kind, raw_path, size=256):
    """Thumbnail for results without a thumbs folder (older batch runs): from the result image or
    the raw image, whichever is asked for, else the other one. Runs in a worker thread, so no
    tkinter here."""
    def result():
        bgr = cv2.imread(str(out_dir / "overlays" / f"{row['image']}.png"))
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def raw():
        return io.load_image(raw_path) if raw_path is not None else None

    first, second = (result, raw) if kind == "Result" else (raw, result)
    img = first()
    if img is None:
        img = second()
    if img is None:
        return None
    im = Image.fromarray(img)
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    return im


class MetricRow(ttk.Frame):
    """Enable box, dataset statistics, pass / fail cutoffs and per-colour counts for one metric."""

    def __init__(self, master, metric, on_change, wrap):
        super().__init__(master, padding=(0, 2, 0, 6))
        self.m, self.on_change = metric, on_change
        decimals = max(0, -int(np.floor(np.log10(metric.step) + 1e-9))) if metric.step < 1 else 0

        self.enabled = tk.BooleanVar(value=metric.enabled)
        ttk.Checkbutton(self, text=metric.label, variable=self.enabled, command=self._toggle,
                        style="Bold.TCheckbutton").grid(row=0, column=0, columnspan=4, sticky="w")
        self.stats_lbl = ttk.Label(self, foreground="#555", font=("Consolas", 8), wraplength=wrap)
        self.stats_lbl.grid(row=1, column=0, columnspan=6, sticky="w", padx=(20, 0))

        self.pass_var = tk.StringVar(value=f"{metric.pass_max:.{decimals}f}")
        self.fail_var = tk.StringVar(value=f"{metric.fail_min:.{decimals}f}")
        for col, (text, var, color) in enumerate([("Pass <=", self.pass_var, COLORS["pass"]),
                                                  ("Fail >", self.fail_var, COLORS["fail"])]):
            tk.Label(self, text=text, foreground=color, font=("Segoe UI", 9, "bold")).grid(
                row=2, column=2 * col, sticky="e", padx=(20 if col == 0 else 8, 2))
            ttk.Spinbox(self, from_=0, to=1e6, increment=metric.step, textvariable=var, width=6,
                        format=f"%.{decimals}f").grid(row=2, column=2 * col + 1, sticky="w")
            var.trace_add("write", self._changed)

        counts = ttk.Frame(self)
        counts.grid(row=0, column=4, columnspan=2, sticky="e")
        self.count_lbls = {}
        for s in ("pass", "warn", "fail", "nodata"):
            lbl = tk.Label(counts, width=3, background=COLORS[s], font=("Segoe UI", 8, "bold"))
            lbl.pack(side="left", padx=1)
            self.count_lbls[s] = lbl
        self.columnconfigure(5, weight=1)

    def _toggle(self):
        self.m.enabled = self.enabled.get()
        self.on_change()

    def _changed(self, *_):
        try:
            p, f = float(self.pass_var.get()), float(self.fail_var.get())
        except ValueError:
            return  # half-typed number
        if (p, f) != (self.m.pass_max, self.m.fail_min):
            self.m.pass_max, self.m.fail_min = p, f
            self.on_change()

    def update_stats(self, devices):
        if devices is None:
            self.stats_lbl.configure(text="no data loaded")
            return
        s = wafer.stats(devices, self.m)
        self.stats_lbl.configure(text=f"min {fmt(s['min'])}  mean {fmt(s['mean'])}  max {fmt(s['max'])}  "
                                      f"median {fmt(s['median'])}  n={s['n']}")
        for status, lbl in self.count_lbls.items():
            lbl.configure(text=str(s[status]))


class WaferMapTab(ttk.Frame):
    """With a `measurement` (measurements.Measurement): that type's results, tolerances and device
    positions. `analysis` (from its config, see analyses.py) picks the default tolerance metrics."""

    def __init__(self, master, measurement=None, analysis="mirror"):
        super().__init__(master)
        self.measurement = measurement
        self.analysis = analysis
        self.pw = int(PANEL_WIDTH * 1.15 * self.winfo_fpixels("1i") / 96)
        self.tolerances_path = measurement.tolerances if measurement else wafer.DEFAULT_TOLERANCES
        self.positions_path = measurement.positions if measurement else wafer.DEFAULT_POSITIONS
        self.default_root = measurement.output if measurement else wafer.ROOT / "output"
        # tunings to name older results by (results.csv without a `tuning` column)
        self.tunings_dir = tuning.tunings_dir(measurement.config) if measurement else None
        self.positions = {}
        self.results_root = None   # folder with one results folder per wafer
        self.wafers = {}           # wafer name -> results folder
        self.out_dir = None        # results folder of the wafer shown
        self.devices = None   # one row per device: position + results
        self.objects = None
        self.metrics = wafer.load_tolerances(self.tolerances_path, analysis)
        self.selected = None
        self.open_in_tuning = None   # set by the measurement tab: fn(image path)
        self.on_show = None          # set by the measurement tab: brings this tab to the front
        self.wafer_var = tk.StringVar()
        self.color_by = tk.StringVar(value=OVERALL)
        self.flip_y = tk.BooleanVar(value=False)
        self.thumb_kind = tk.StringVar(value="Raw")
        self.metric_rows = []
        self._tip = None
        self._windows = {}
        self.stats_window = None    # the type's statistics tab (gui_stats.StatsPanel), once built
        # thumbnails: PIL images per (folder, image, kind), tk images per box size, and a worker
        # thread that makes the ones a batch run didn't save
        self._thumbs = {}
        self._thumbs_read = self._thumbs_total = 0   # thumbnails read from disk / shown by draw(), for the status bar
        self._photos = {}
        self._photo_size = None
        self._pending = set()
        self._todo, self._made = queue.Queue(), queue.Queue()
        threading.Thread(target=self._thumb_worker, daemon=True).start()

        style = ttk.Style(self)
        dpi = self.winfo_fpixels("1i") / 96
        style.configure("Treeview", rowheight=int(20 * dpi))
        style.configure("Bold.TCheckbutton", font=("Segoe UI", 9, "bold"))

        self._build()
        self._load_positions(self.positions_path, quiet=True)
        self.set_results_root(self.default_root, quiet=True)

    # ------------------------------------------------------------ layout

    def _build(self):
        right = ttk.Frame(self, padding=8, width=self.pw)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)   # keep the fixed width
        ttk.Separator(self, orient="vertical").pack(side="right", fill="y")

        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)
        top = ttk.Frame(left, padding=(8, 6))
        top.pack(fill="x")
        self.title_lbl = ttk.Label(top, font=("Segoe UI", 12, "bold"))
        self.title_lbl.pack(side="left")
        self.tuning_lbl = ttk.Label(top, foreground="#666", font=("Segoe UI", 9))
        self.tuning_lbl.pack(side="left", padx=(12, 0))
        self.legend = ttk.Frame(top)
        self.legend.pack(side="right")
        self.canvas = tk.Canvas(left, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())

        data = ttk.LabelFrame(right, text="Data", padding=6)
        data.pack(fill="x")
        btns = ttk.Frame(data)
        btns.pack(fill="x")
        ttk.Button(btns, text="Results folder...", command=self._ask_results_root).pack(side="left")
        ttk.Button(btns, text="Position map...", command=self._ask_positions).pack(side="left", padx=6)
        pick = ttk.Frame(data)
        pick.pack(fill="x", pady=(6, 0))
        ttk.Label(pick, text="Wafer:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Button(pick, text=">", width=3, command=lambda: self._step_wafer(1)).pack(side="right")
        ttk.Button(pick, text="<", width=3, command=lambda: self._step_wafer(-1)).pack(side="right", padx=2)
        self.wafer_box = ttk.Combobox(pick, textvariable=self.wafer_var, state="readonly")
        self.wafer_box.pack(side="left", fill="x", expand=True, padx=4)
        self.wafer_box.bind("<<ComboboxSelected>>", lambda e: self._wafer_chosen())
        self.data_lbl = ttk.Label(data, wraplength=self.pw - 40, justify="left", foreground="#444")
        self.data_lbl.pack(fill="x", pady=(4, 0))

        view = ttk.Frame(right)
        view.pack(fill="x", pady=(8, 0))
        ttk.Label(view, text="Colour by:").grid(row=0, column=0, sticky="w")
        self.color_box = ttk.Combobox(view, textvariable=self.color_by, state="readonly", width=30)
        self.color_box.grid(row=0, column=1, sticky="ew", padx=4)
        self.color_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_map())
        ttk.Label(view, text="Thumbnail:").grid(row=1, column=0, sticky="w", pady=(2, 0))
        kinds = ttk.Frame(view)
        kinds.grid(row=1, column=1, sticky="w", padx=4, pady=(2, 0))
        for kind in THUMBS:
            ttk.Radiobutton(kinds, text=kind, value=kind, variable=self.thumb_kind,
                            command=self.draw).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(view, text="Flip Y (Y up)", variable=self.flip_y,
                        command=self.draw).grid(row=2, column=1, sticky="w", padx=4, pady=(2, 0))
        view.columnconfigure(1, weight=1)

        sel = ttk.LabelFrame(right, text="Selected device", padding=6)
        sel.pack(side="bottom", fill="x", pady=(8, 0))
        self.sel_text = tk.Text(sel, width=40, height=6, font=("Consolas", 9), relief="flat",
                                background="#f4f4f4", state="disabled", wrap="none")
        self.sel_text.pack(fill="x")
        for s in COLORS:
            self.sel_text.tag_configure(s, background=LIGHT[s])
        self.sel_text.tag_configure("head", font=("Consolas", 9, "bold"))
        self.open_btn = ttk.Button(sel, text="Open device...", command=self.open_device, state="disabled")
        self.open_btn.pack(anchor="e", pady=(4, 0))

        head = ttk.Frame(right)
        head.pack(fill="x", pady=(10, 2))
        ttk.Label(head, text="Tolerances", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Reset to defaults", command=self._reset_tolerances).pack(side="right")
        ttk.Button(head, text="Save", command=self._save_tolerances).pack(side="right", padx=4)
        ttk.Label(right, text="Counts per metric:  pass / warn / fail / no data", foreground="#666",
                  font=("Segoe UI", 8)).pack(anchor="w")
        self.tol = ScrollFrame(right)
        self.tol.pack(fill="both", expand=True, pady=(2, 0))
        self._build_metric_rows()

    def _build_metric_rows(self):
        self.tol.clear()
        self.metric_rows = []
        for m in self.metrics:
            row = MetricRow(self.tol.inner, m, self._refresh_map, self.pw - 110)
            row.pack(fill="x", padx=(0, 6))
            self.metric_rows.append(row)
        labels = [OVERALL] + [m.label for m in self.metrics]
        self.color_box.configure(values=labels)
        if self.color_by.get() not in labels:
            self.color_by.set(OVERALL)

    # ------------------------------------------------------------ data

    def _ask_positions(self):
        p = filedialog.askopenfilename(initialdir=self.positions_path.parent,
                                       filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if p:
            self._load_positions(Path(p))

    def _load_positions(self, path, quiet=False):
        try:
            self.positions = wafer.load_positions(path)
            self.positions_path = Path(path)
        except Exception as e:
            self.positions = {}
            if not quiet:
                messagebox.showerror("Position map", f"Could not read {path}:\n{e}")
        if self.out_dir is not None:
            self.load_results(self.out_dir, quiet=quiet)
        else:
            self._refresh()

    def _ask_results_root(self):
        start = self.results_root or self.default_root
        root = filedialog.askdirectory(title="Batch output folder (one subfolder per wafer, or one results.csv)",
                                       initialdir=start)
        if root:
            self.set_results_root(Path(root))

    def set_results_root(self, root, quiet=False):
        """Find the wafers in a batch output folder and show one (the one shown before, if it is
        there; otherwise the first)."""
        root = Path(root)
        with loading(f"Looking for results in {root}..."):
            found = wafer.find_results(root)
        if not found:
            if not quiet:
                messagebox.showerror("Load results", f"No results.csv in {root} or its subfolders.")
            if self.devices is None:
                self.results_root = root
                self._refresh()
            return
        self.results_root, self.wafers = root, found
        if self._stats_open():
            self.stats_window.reload()
        self.wafer_box.configure(values=list(found))
        name = self.wafer_var.get()
        self.wafer_var.set(name if name in found else next(iter(found)))
        self._wafer_chosen(quiet)

    def _step_wafer(self, d):
        names = list(self.wafers)
        if names:
            i = names.index(self.wafer_var.get()) if self.wafer_var.get() in names else 0
            self.wafer_var.set(names[(i + d) % len(names)])
            self._wafer_chosen()

    def _wafer_chosen(self, quiet=False):
        folder = self.wafers.get(self.wafer_var.get())
        if folder is not None:
            self.load_results(folder, quiet)

    def load_results(self, out_dir, quiet=False):
        try:
            with loading(f"Loading the results of {Path(out_dir).name}..."):
                self.devices, self.objects = wafer.load_results(out_dir, self.positions, self.tunings_dir)
            self.out_dir = Path(out_dir)
        except Exception as e:
            if not quiet:
                messagebox.showerror("Load results", f"Could not load {out_dir}:\n{e}")
            return
        self.selected = None
        self._refresh()

    def _metric(self, label):
        return next((m for m in self.metrics if m.label == label), None)

    def status_of(self, row):
        m = self._metric(self.color_by.get())
        return wafer.overall(row, self.metrics) if m is None else m.status(m.value(row))

    def _records(self):
        if self.devices is None:
            return []
        return self.devices.to_dict("records")

    # ------------------------------------------------------------ refresh

    def _refresh(self):
        if self.devices is None:
            pos = f"{len(self.positions)} positions in {self.positions_path.name}" if self.positions else \
                f"No position map loaded ({self.positions_path.name})"
            where = f" in {self.results_root}" if self.results_root else ""
            self.data_lbl.configure(text=f"{pos}\nNo results{where} yet: run the batch from the "
                                         f"step-by-step tuning tab ('Run on all images...'), or 'Results folder...'.")
        else:
            d = self.devices
            has_res = d["image"].notna()
            on_map = d["x"].notna()
            where = (f"{len(self.positions)} positions in {self.positions_path.name}" if self.positions else
                     f"no position map {self.positions_path.name} yet: shown in a row by number")
            text = (f"Tuning: {wafer.tuning_of(d)}\n"
                    f"Results: {self.out_dir}\n"
                    f"{int((has_res & on_map).sum())} devices with results on the map ({where})")
            missing = d.loc[on_map & ~has_res, "device"].tolist()
            off_map = d.loc[has_res & ~on_map, "device"].tolist()
            if missing:
                text += f"\nNo image for devices: {', '.join(map(str, missing))}"
            if off_map:
                text += f"\nNot on the map: devices {', '.join(map(str, off_map))}"
            self.data_lbl.configure(text=text)
        self._refresh_map()

    def _refresh_map(self):
        for row in self.metric_rows:
            row.update_stats(self.devices)
        self.draw()
        self._show_selected()
        if self._stats_open():
            self.stats_window.redraw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        self._hide_tip()
        for w in self.legend.winfo_children():
            w.destroy()
        recs = [r for r in self._records() if not _isnan(r["x"]) and not _isnan(r["y"])]
        self._thumbs_read, self._thumbs_total = 0, len(recs)
        cw, ch = c.winfo_width(), c.winfo_height()
        name = self.wafer_var.get()
        self.title_lbl.configure(text=f"{name}  -  {self.color_by.get()}" if name else self.color_by.get())
        self.tuning_lbl.configure(text=f"tuning: {wafer.tuning_of(self.devices)}" if self.devices is not None else "")
        if not recs:
            c.create_text(cw / 2, ch / 2, text="No devices to show: load a position map and results.",
                          fill="#888", font=("Segoe UI", 12))
            return

        xs = np.array([r["x"] for r in recs])
        ys = np.array([r["y"] for r in recs])
        px, py = _pitch(xs), _pitch(ys)
        span_x, span_y = xs.max() - xs.min() + px, ys.max() - ys.min() + py
        margin = 20
        s = min((cw - 2 * margin) / span_x, (ch - 2 * margin) / span_y)
        ox, oy = (cw - span_x * s) / 2, (ch - span_y * s) / 2
        w, h = px * s * 0.46, py * s * 0.46
        kind = self.thumb_kind.get()
        thumbs = THUMBS.get(kind) is not None
        big = -max(9, int(min(w, h) * 0.6))     # negative = pixels, so the text scales with the boxes
        small = -max(8, int(min(w, h) * 0.32))
        border = max(3, round(min(w, h) * 0.1))          # coloured frame around a thumbnail
        strip = max(14, round(h * 0.34))                  # coloured bottom strip with the numbers
        strip_font = -max(8, int(strip * 0.68))

        counts = {k: 0 for k in COLORS}
        metric = self._metric(self.color_by.get())
        for r in recs:
            dev = int(r["device"])
            gx = ox + (r["x"] - xs.min() + px / 2) * s
            gy = oy + ((ys.max() - r["y"]) if self.flip_y.get() else (r["y"] - ys.min())) * s + py / 2 * s
            st = "nodata" if _isnan(r.get("image", np.nan)) else self.status_of(r)
            counts[st] += 1
            tag = f"dev{dev}"
            selected = dev == self.selected
            if metric is None:
                nf = sum(m.enabled and m.status(m.value(r)) == "fail" for m in self.metrics)
                nw = sum(m.enabled and m.status(m.value(r)) == "warn" for m in self.metrics)
                sub = "no data" if st == "nodata" else (" ".join(
                    x for x in (f"{nf} fail" if nf else "", f"{nw} warn" if nw else "") if x) or "OK")
            else:
                sub = fmt(r.get(metric.column, np.nan))
            x0, y0, x1, y1 = gx - w, gy - h, gx + w, gy + h
            if selected:
                c.create_rectangle(x0 - 4, y0 - 4, x1 + 4, y1 + 4, outline="#0d47a1", width=4, tags=(tag,))
            if thumbs:
                # status-coloured frame, the device image inside, numbers in the bottom strip
                c.create_rectangle(x0, y0, x1, y1, fill=COLORS[st], outline="#555", tags=(tag,))
                ix0, iy0, ix1, iy1 = x0 + border, y0 + border, x1 - border, y1 - strip
                c.create_rectangle(ix0, iy0, ix1, iy1, fill="#2b2b2b", outline="", tags=(tag,))
                photo = None if st == "nodata" else self._photo(r, kind, int(ix1 - ix0), int(iy1 - iy0))
                if photo is not None:
                    c.create_image((ix0 + ix1) / 2, (iy0 + iy1) / 2, image=photo, tags=(tag,))
                else:
                    pending = (str(self.out_dir), r.get("image"), kind) in self._pending
                    c.create_text((ix0 + ix1) / 2, (iy0 + iy1) / 2, fill="#aaa", font=("Segoe UI", small),
                                  text="loading..." if pending else "no image", tags=(tag,))
                c.create_text(gx, y1 - strip / 2, text=f"{dev}   {sub}", fill=TEXT_ON[st],
                              font=("Segoe UI", strip_font, "bold"), tags=(tag,))
            else:
                self._round_rect(x0, y0, x1, y1, min(w, h) * 0.25, fill=COLORS[st], outline="#555", tags=(tag,))
                c.create_text(gx, gy - h * 0.3, text=str(dev), font=("Segoe UI", big, "bold"), tags=(tag,))
                c.create_text(gx, gy + h * 0.4, text=sub, font=("Segoe UI", small), tags=(tag,))
            c.tag_bind(tag, "<Button-1>", lambda e, d=dev: self.select(d))
            c.tag_bind(tag, "<Double-Button-1>", lambda e, d=dev: (self.select(d), self.open_device()))
            c.tag_bind(tag, "<Enter>", lambda e, rec=r: self._show_tip(e, rec))
            c.tag_bind(tag, "<Leave>", lambda e: self._hide_tip())

        for st in ("pass", "warn", "fail", "nodata"):
            tk.Label(self.legend, text=f" {NAMES[st]}: {counts[st]} ", background=COLORS[st],
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=2)
        if self._thumbs_read:
            ready()

    def _round_rect(self, x0, y0, x1, y1, r, **kw):
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
               x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    # ------------------------------------------------------------ thumbnails

    def _photo(self, rec, kind, width, height):
        """The device's thumbnail scaled to fit width x height, or None while it isn't there."""
        key = (str(self.out_dir), rec["image"], kind)
        base = self._thumb(key, rec, kind)
        if base is None or width < 4 or height < 4:
            return None
        if self._photo_size != (width, height):     # boxes changed size: rescale everything
            self._photos, self._photo_size = {}, (width, height)
        if key not in self._photos:
            f = min(width / base.width, height / base.height)
            im = base.resize((max(1, round(base.width * f)), max(1, round(base.height * f))),
                             Image.Resampling.LANCZOS)
            self._photos[key] = ImageTk.PhotoImage(im)
        return self._photos[key]

    def _thumb(self, key, rec, kind):
        """PIL thumbnail from the batch run's thumbs folder; if it has none (older runs), one is
        made in the background and the map is redrawn when it's ready."""
        if key in self._thumbs:
            return self._thumbs[key]
        path = self.out_dir / THUMBS[kind] / f"{rec['image']}.jpg"
        if path.exists():
            self._thumbs_read += 1
            status(f"Loading thumbnail {self._thumbs_read} of {self._thumbs_total} ({rec['image']})...")
            try:
                im = Image.open(path).convert("RGB")
            except OSError:
                im = None
            self._thumbs[key] = im
            return im
        if key not in self._pending:
            if not self._pending:
                self.after(150, self._poll_thumbs)
            self._pending.add(key)
            self._todo.put((key, self.out_dir, rec, kind, self.raw_image_path(rec)))
        return None

    def _thumb_worker(self):
        while True:
            key, out_dir, rec, kind, raw = self._todo.get()
            try:
                im = _make_thumb(out_dir, rec, kind, raw)
            except Exception:
                im = None
            self._made.put((key, im))

    def _poll_thumbs(self):
        got = False
        while True:
            try:
                key, im = self._made.get_nowait()
            except queue.Empty:
                break
            self._thumbs[key] = im
            self._pending.discard(key)
            got = True
        if got:
            self.draw()
        if self._pending:
            self.after(150, self._poll_thumbs)

    # ------------------------------------------------------------ selection, tooltip

    def _row(self, dev):
        if self.devices is None:
            return None
        rows = self.devices[self.devices["device"] == dev]
        return rows.iloc[0].to_dict() if len(rows) else None

    def select(self, dev):
        self.selected = dev
        self.draw()
        self._show_selected()

    def _show_selected(self):
        t = self.sel_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        row = self._row(self.selected) if self.selected is not None else None
        if row is None:
            t.insert("end", "Click a device on the map.\nDouble-click opens all its data.")
            t.configure(state="disabled", height=3)
            self.open_btn.configure(state="disabled")
            return
        st = wafer.overall(row, self.metrics)
        t.insert("end", f"Device {int(row['device'])}   {fmt(row.get('image'))}\n", "head")
        t.insert("end", f"Overall: {NAMES[st].upper()}\n", st)
        for m in self.metrics:
            v = m.value(row)
            s = m.status(v)
            mark = "" if m.enabled else " (off)"
            t.insert("end", f"{m.label[:22]:22s} {fmt(row.get(m.column, np.nan)):>7s} {NAMES[s]}{mark}\n", s)
        t.configure(state="disabled", height=min(16, len(self.metrics) + 3))
        self.open_btn.configure(state="normal" if not _isnan(row.get("image", np.nan)) else "disabled")

    def _show_tip(self, e, rec):
        self._hide_tip()
        lines = [f"Device {int(rec['device'])}   ({fmt(rec['x'])}, {fmt(rec['y'])})", fmt(rec.get("image"))]
        for m in self.metrics:
            if m.enabled:
                lines.append(f"{m.label}: {fmt(rec.get(m.column, np.nan))}  [{NAMES[m.status(m.value(rec))]}]")
        tip = tk.Toplevel(self)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{e.x_root + 16}+{e.y_root + 12}")
        tk.Label(tip, text="\n".join(lines), justify="left", background="#ffffe8", relief="solid",
                 borderwidth=1, font=("Consolas", 8), padx=4, pady=2).pack()
        self._tip = tip

    def _hide_tip(self):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    # ------------------------------------------------------------ actions

    def open_device(self):
        row = self._row(self.selected) if self.selected is not None else None
        if row is None or _isnan(row.get("image", np.nan)):
            return
        key = (str(self.out_dir), self.selected)
        win = self._windows.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        objs = None
        if self.objects is not None:
            objs = self.objects[self.objects["image"] == row["image"]]
        self._windows[key] = DeviceWindow(self, row, objs)

    def _stats_open(self):
        return self.stats_window is not None and self.stats_window.winfo_exists()

    def show_device(self, row):
        """From the statistics tab (a row of wafer.load_all): show that wafer on the map with the
        device selected, and open the device window."""
        name = row["wafer"]
        if name in self.wafers and Path(row["out_dir"]) == self.wafers[name]:
            if self.wafer_var.get() != name:
                self.wafer_var.set(name)
                self._wafer_chosen()
            self.select(int(row["device"]))
            if self.on_show:
                self.on_show()
        self.open_device_from(row)

    def open_device_from(self, row):
        """Device window for a row of wafer.load_all (any wafer, not just the one on the map)."""
        out_dir = Path(row["out_dir"])
        key = (str(out_dir), int(row["device"]))
        win = self._windows.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        objs = None
        if (out_dir / "objects.csv").exists():
            with loading(f"Loading the objects of {row['wafer']}..."):
                objs = pd.read_csv(out_dir / "objects.csv", keep_default_na=False, na_values=[""])
            objs = objs[objs["image"] == row["image"]]
        self._windows[key] = DeviceWindow(self, row, objs, out_dir, row["wafer"])

    def _save_tolerances(self):
        try:
            wafer.save_tolerances(self.metrics, self.tolerances_path)
        except OSError as e:
            messagebox.showerror("Save tolerances", str(e))
            return
        messagebox.showinfo("Save tolerances", f"Saved to {self.tolerances_path}")

    def _reset_tolerances(self):
        self.metrics = wafer.default_metrics(self.analysis)
        self._build_metric_rows()
        self._refresh_map()

    def raw_image_path(self, row):
        p = row.get("image_path")
        if isinstance(p, str) and workspace.resolve(p).exists():
            return workspace.resolve(p)
        for suffix in (".tif", ".tiff"):
            q = wafer.ROOT / "Images" / f"{row['image']}{suffix}"
            if q.exists():
                return q
        return None


class DeviceWindow(tk.Toplevel):
    """Everything about one device: result image / raw image, all results, all detected objects."""

    def __init__(self, tab, row, objects, out_dir=None, wafer_name=None):
        super().__init__(tab)
        self.tab, self.row = tab, row
        self.objects = objects
        # stays with this wafer when the map shows another one
        self.out_dir = Path(out_dir) if out_dir is not None else tab.out_dir
        self.wafer = wafer_name or tab.wafer_var.get()
        self.dev = int(row["device"])
        self.title(f"{self.wafer} - device {self.dev} - {row['image']}")
        dpi = self.winfo_fpixels("1i") / 96
        w = min(int(1300 * dpi), int(self.winfo_screenwidth() * 0.9))
        h = min(int(850 * dpi), int(self.winfo_screenheight() * 0.85))
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth() - w) // 2}+{(self.winfo_screenheight() - h) // 3}")
        self.px = float(row.get("um_per_px") or 2.51)
        self._images = {}
        self._crop_photo = None

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        body.add(left, weight=1)
        bar = ttk.Frame(left, padding=(6, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="Show:").pack(side="left")
        self.view_var = tk.StringVar(value="Result (outlined)")
        box = ttk.Combobox(bar, textvariable=self.view_var, state="readonly", width=20,
                           values=["Result (outlined)", "Raw image"])
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", lambda e: self._show_image())
        ttk.Button(bar, text="Fit", command=lambda: self.view.fit()).pack(side="left", padx=4)
        ttk.Button(bar, text="Open in step-by-step tuning", command=self._open_tuning).pack(side="right")
        self.view = ImageView(left)
        self.view.pack(fill="both", expand=True)

        right = ttk.PanedWindow(body, orient="vertical", width=int(560 * dpi))
        body.add(right, weight=0)
        right.add(self._build_results(right), weight=1)
        right.add(self._build_objects(right), weight=1)

        self.after(50, self._show_image)

    # ------------------------------------------------------------ panels

    def _build_results(self, master):
        frame = ttk.LabelFrame(master, text=f"{self.wafer}, device {self.dev}: results", padding=4)
        tree = ttk.Treeview(frame, columns=("value", "status"), show="tree headings")
        tree.heading("#0", text="Measure")
        tree.heading("value", text="Value")
        tree.heading("status", text="Status")
        tree.column("#0", width=230)
        tree.column("value", width=110, anchor="e")
        tree.column("status", width=90)
        for s in COLORS:
            tree.tag_configure(s, background=LIGHT[s])
        tree.tag_configure("group", font=("Segoe UI", 9, "bold"))

        st = wafer.overall(self.row, self.tab.metrics)
        tol = tree.insert("", "end", text="Tolerances", values=("", NAMES[st].upper()), open=True, tags=(st, "group"))
        for m in self.tab.metrics:
            s = m.status(m.value(self.row))
            tree.insert(tol, "end", text=m.label, values=(fmt(self.row.get(m.column, np.nan)),
                                                         NAMES[s] + ("" if m.enabled else " (off)")), tags=(s,))
        allv = tree.insert("", "end", text="All values", open=True, tags=("group",))
        for k, v in self.row.items():
            tree.insert(allv, "end", text=k, values=(fmt(v), ""))
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        return frame

    def _build_objects(self, master):
        n = 0 if self.objects is None else len(self.objects)
        frame = ttk.LabelFrame(master, text=f"Detected objects ({n})  -  click: mark, double-click: zoom", padding=4)
        if self.objects is None:
            ttk.Label(frame, text="No objects.csv in the results folder.").pack()
            return frame

        top = ttk.Frame(frame)
        top.pack(fill="both", expand=True)
        cols = [c for c, _, _ in OBJECT_COLUMNS]
        self.obj_tree = tree = ttk.Treeview(top, columns=cols, show="headings", selectmode="browse")
        for c, text, width in OBJECT_COLUMNS:
            tree.heading(c, text=text, command=lambda c=c: self._sort(c))
            tree.column(c, width=width, anchor="w" if c in ("class", "label") else "e")
        for cls, bgr in report.CLASS_COLORS.items():
            tree.tag_configure(cls, foreground=_bgr_hex(tuple(int(0.75 * v) for v in bgr)))
        self._obj_rows = {}
        for o in self.objects.to_dict("records"):
            iid = str(o["object_id"])
            self._obj_rows[iid] = o
            tree.insert("", "end", iid=iid, values=[fmt(o.get(c, np.nan)) if not _isnan(o.get(c, np.nan))
                                                    else "" for c in cols], tags=(o.get("class", ""),))
        bar = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        tree.bind("<<TreeviewSelect>>", lambda e: self._object_selected(zoom=False))
        tree.bind("<Double-Button-1>", lambda e: self._object_selected(zoom=True))

        bottom = ttk.Frame(frame)
        bottom.pack(fill="x", pady=(4, 0))
        self.crop_lbl = ttk.Label(bottom)
        self.crop_lbl.pack(side="left")
        self.obj_info = ttk.Label(bottom, justify="left", font=("Consolas", 8), foreground="#444")
        self.obj_info.pack(side="left", padx=8, anchor="n")
        self._sort_state = {}
        return frame

    def _sort(self, col):
        desc = self._sort_state.get(col, False)
        items = list(self._obj_rows.items())

        def key(item):
            v = item[1].get(col, np.nan)
            if isinstance(v, (int, float, np.integer, np.floating)):
                return (0, -np.inf if _isnan(v) else float(v), "")
            return (1, 0.0, str(v))

        items.sort(key=key, reverse=desc)
        for i, (iid, _) in enumerate(items):
            self.obj_tree.move(iid, "", i)
        self._sort_state[col] = not desc

    # ------------------------------------------------------------ actions

    def _image(self, which):
        if which not in self._images:
            with loading(f"Opening {which.lower()} image of {self.row['image']}..."):
                self._images[which] = self._load_image(which)
        return self._images[which]

    def _load_image(self, which):
        img = None
        if which == "Result (outlined)" and self.out_dir is not None:
            bgr = cv2.imread(str(self.out_dir / "overlays" / f"{self.row['image']}.png"))
            img = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if img is None:
            path = self.tab.raw_image_path(self.row)
            img = io.load_image(path) if path else None
        return img

    def _show_image(self):
        img = self._image(self.view_var.get())
        if img is None:
            messagebox.showwarning("Device", "Image not found.", parent=self)
            return
        self.view.set_image(img)

    def _object_selected(self, zoom):
        sel = self.obj_tree.selection()
        if not sel:
            return
        o = self._obj_rows[sel[0]]
        size_um = max(float(o.get("length_um") or 0), float(o.get("eq_diam_um") or 0))
        radius = size_um / self.px / 2 + 6
        if zoom:
            self.view.zoom_to(o["cx"], o["cy"], max(self.view.scale, 4.0), radius)
        else:
            self.view.marker = (o["cx"], o["cy"], radius)
            self.view.render()
        self.obj_info.configure(text="\n".join(f"{k}: {fmt(o.get(k))}" for k in (
            "object_id", "class", "area_um2", "eq_diam_um", "length_um", "elongation", "solidity",
            "mean_ratio", "edge_dist_um", "in_inner")))
        crop = self.out_dir / "crops" / f"{o['object_id']}.png" if self.out_dir else None
        if crop is not None and crop.exists():
            im = Image.open(crop)
            self._crop_photo = ImageTk.PhotoImage(im.resize((im.width * 2, im.height * 2), Image.Resampling.NEAREST))
            self.crop_lbl.configure(image=self._crop_photo)
        else:
            self.crop_lbl.configure(image="", text="(no crop)")

    def _open_tuning(self):
        path = self.tab.raw_image_path(self.row)
        if path is None or self.tab.open_in_tuning is None:
            messagebox.showwarning("Device", "Raw image not found.", parent=self)
            return
        self.tab.open_in_tuning(path)
        self.tab.winfo_toplevel().lift()
