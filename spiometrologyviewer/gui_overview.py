"""Wafer overview tab: the raw images of a measurement type without an analysis (config `analysis:
none` with `view: wafer`, e.g. the Full Stack stereo microscope images), one thumbnail per device in
its position on the wafer. A click selects a device, a double-click opens its images at full
resolution. No results, tolerances or batch run.

The device number comes from the file name (config `input.device_pattern`, a regex whose first group
is the number; default wafer.device_number). A device can have several images (U06D_1, U06D_2).
Thumbnails are made in a background thread and cached in the type's output folder
(<output>/<wafer>/thumbs/<image>.jpg), so the next visit doesn't read the full images again."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from . import io, measurements, wafer
from .gui import PANEL_WIDTH, ImageView, loading, ready, status
from .gui_wafer import _pitch

FRAME = "#9e9e9e"       # frame of a device with images
EMPTY = "#e0e0e0"       # ... without
STRIP_TEXT = "white"


def _thumb_path(cache_dir, image):
    return Path(cache_dir) / f"{Path(image).stem}.jpg"


def make_thumb(image, cache_dir, size):
    """PIL thumbnail of `image`: the cached one if it is newer than the image, else made from the
    image and cached (if the cache folder can be written). Runs in a worker thread: no tkinter."""
    image = Path(image)
    cached = _thumb_path(cache_dir, image) if cache_dir is not None else None
    try:
        if cached is not None and cached.exists() and cached.stat().st_mtime >= image.stat().st_mtime:
            with Image.open(cached) as im:
                return im.convert("RGB")
    except OSError:
        pass
    im = Image.fromarray(io.load_image(image))
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            im.save(cached, quality=90)
        except OSError:
            pass   # read-only output folder: just not cached
    return im


class WaferOverviewTab(ttk.Frame):
    def __init__(self, master, measurement, cfg=None):
        super().__init__(master)
        self.m = measurement
        self.cfg = cfg or measurement.load_config()
        inp = self.cfg.get("input", {})
        self.pattern = inp.get("device_pattern")
        self.thumb_px = int(self.cfg.get("output", {}).get("thumb_px", 256))
        self.pw = int(PANEL_WIDTH * self.winfo_fpixels("1i") / 96)
        self.positions_path = measurement.positions
        self.positions = {}
        self.wafers = {}          # label -> image folder
        self.folder = None        # image folder of the wafer shown
        self.devices = {}         # device -> [image paths]
        self.unnamed = []         # images without a device number
        self.selected = None
        self.wafer_var = tk.StringVar()
        self.flip_y = tk.BooleanVar(value=False)
        self._tip = None
        self._windows = {}
        # thumbnails: PIL images per image path, tk images per box size, a worker making the missing ones
        self._thumbs, self._photos, self._photo_size = {}, {}, None
        self._pending = set()
        self._todo, self._made = queue.Queue(), queue.Queue()
        threading.Thread(target=self._thumb_worker, daemon=True).start()

        self._build()
        self._load_positions(self.positions_path, quiet=True)
        self.refresh()

    # ------------------------------------------------------------ layout

    def _build(self):
        right = ttk.Frame(self, padding=8, width=self.pw)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ttk.Separator(self, orient="vertical").pack(side="right", fill="y")

        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)
        top = ttk.Frame(left, padding=(8, 6))
        top.pack(fill="x")
        self.title_lbl = ttk.Label(top, font=("Segoe UI", 12, "bold"))
        self.title_lbl.pack(side="left")
        self.count_lbl = ttk.Label(top, foreground="#666", font=("Segoe UI", 9))
        self.count_lbl.pack(side="left", padx=(12, 0))
        self.canvas = tk.Canvas(left, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())

        data = ttk.LabelFrame(right, text="Data", padding=6)
        data.pack(fill="x")
        btns = ttk.Frame(data)
        btns.pack(fill="x")
        ttk.Button(btns, text="Refresh", command=self.refresh).pack(side="left")
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
        ttk.Checkbutton(right, text="Flip Y (Y up)", variable=self.flip_y,
                        command=self.draw).pack(anchor="w", pady=(8, 0))
        ttk.Label(right, text=f"{self.m.key}: no analysis yet, the images can only be looked at.",
                  foreground="#666", wraplength=self.pw - 30, justify="left").pack(anchor="w", pady=(8, 0))

        sel = ttk.LabelFrame(right, text="Selected device", padding=6)
        sel.pack(side="bottom", fill="x", pady=(8, 0))
        self.sel_lbl = ttk.Label(sel, text="", font=("Consolas", 9), justify="left", wraplength=self.pw - 40)
        self.sel_lbl.pack(fill="x")
        self.open_btn = ttk.Button(sel, text="Open image...", command=self.open_device, state="disabled")
        self.open_btn.pack(anchor="e", pady=(4, 0))

    # ------------------------------------------------------------ data

    def _ask_positions(self):
        p = filedialog.askopenfilename(initialdir=self.positions_path.parent,
                                       filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if p:
            self._load_positions(Path(p))
            self._refresh()

    def _load_positions(self, path, quiet=False):
        try:
            self.positions = wafer.load_positions(path)
            self.positions_path = Path(path)
        except Exception as e:
            self.positions = {}
            if not quiet:
                messagebox.showerror("Position map", f"Could not read {path}:\n{e}")

    def refresh(self):
        with loading(f"Looking for wafers of {self.m.key}..."):
            self.wafers = self.m.browsable(self.cfg)
        self.wafer_box.configure(values=list(self.wafers))
        if not self.wafers:
            self.folder, self.devices, self.unnamed = None, {}, []
            self._refresh()
            return
        if self.wafer_var.get() not in self.wafers:
            self.wafer_var.set(next(iter(self.wafers)))
        self._wafer_chosen()

    def _step_wafer(self, d):
        names = list(self.wafers)
        if names:
            i = names.index(self.wafer_var.get()) if self.wafer_var.get() in names else 0
            self.wafer_var.set(names[(i + d) % len(names)])
            self._wafer_chosen()

    def _wafer_chosen(self):
        folder = self.wafers.get(self.wafer_var.get())
        if folder is None:
            return
        with loading(f"Looking for images in {folder}..."):
            images = self.m.images_of(folder, self.cfg)
        self.folder = folder
        self.devices, self.unnamed = {}, []
        for p in images:
            dev = wafer.device_number(p.name, self.pattern)
            if dev is None:
                self.unnamed.append(p)
            else:
                self.devices.setdefault(dev, []).append(p)
        self.selected = None
        self._refresh()

    def _cache_dir(self):
        name = self.wafer_var.get().replace(measurements.LEGACY_LABEL, "")
        return self.m.output / name / "thumbs" if self.folder is not None else None

    def _layout(self):
        """{device: (x, y)} of every device to draw: the position map's (devices with images that
        aren't on it are listed in the data panel), or without a map the devices in a row by number."""
        if not self.positions:
            return {d: (float(d), 0.0) for d in sorted(self.devices)}
        return dict(self.positions)

    def _refresh(self):
        if self.folder is None:
            self.data_lbl.configure(text=f"No wafer folders with images in {self.m.base / self.m.images}")
        else:
            n = sum(len(v) for v in self.devices.values())
            where = (f"{len(self.positions)} positions in {self.positions_path.name}" if self.positions else
                     f"no position map {self.positions_path.name}: shown in a row by number")
            text = f"Images: {self.folder}\n{n} images of {len(self.devices)} devices ({where})"
            if self.positions:
                missing = sorted(set(self.positions) - set(self.devices))
                off_map = sorted(set(self.devices) - set(self.positions))
                if missing:
                    text += f"\nNo image for devices: {', '.join(map(str, missing))}"
                if off_map:
                    text += f"\nNot on the map: devices {', '.join(map(str, off_map))}"
            if self.unnamed:
                text += f"\nNo device number in: {', '.join(p.name for p in self.unnamed)}"
            self.data_lbl.configure(text=text)
        self.draw()
        self._show_selected()

    # ------------------------------------------------------------ drawing

    def draw(self):
        c = self.canvas
        c.delete("all")
        self._hide_tip()
        cw, ch = c.winfo_width(), c.winfo_height()
        name = self.wafer_var.get()
        self.title_lbl.configure(text=name)
        layout = self._layout()
        n_img = sum(len(v) for v in self.devices.values())
        self.count_lbl.configure(text=f"{len(self.devices)} devices, {n_img} images" if self.folder else "")
        if not layout:
            c.create_text(cw / 2, ch / 2, text="No devices to show.", fill="#888", font=("Segoe UI", 12))
            return

        xs = np.array([x for x, _ in layout.values()])
        ys = np.array([y for _, y in layout.values()])
        px, py = _pitch(xs), _pitch(ys)
        span_x, span_y = xs.max() - xs.min() + px, ys.max() - ys.min() + py
        margin = 20
        s = min((cw - 2 * margin) / span_x, (ch - 2 * margin) / span_y)
        ox, oy = (cw - span_x * s) / 2, (ch - span_y * s) / 2
        w, h = px * s * 0.46, py * s * 0.46
        small = -max(8, int(min(w, h) * 0.32))
        border = max(3, round(min(w, h) * 0.06))
        strip = max(14, round(h * 0.3))
        strip_font = -max(8, int(strip * 0.62))

        for dev, (x, y) in sorted(layout.items()):
            images = self.devices.get(dev, [])
            gx = ox + (x - xs.min() + px / 2) * s
            gy = oy + ((ys.max() - y) if self.flip_y.get() else (y - ys.min())) * s + py / 2 * s
            tag = f"dev{dev}"
            x0, y0, x1, y1 = gx - w, gy - h, gx + w, gy + h
            if dev == self.selected:
                c.create_rectangle(x0 - 4, y0 - 4, x1 + 4, y1 + 4, outline="#0d47a1", width=4, tags=(tag,))
            c.create_rectangle(x0, y0, x1, y1, fill=FRAME if images else EMPTY, outline="#555", tags=(tag,))
            ix0, iy0, ix1, iy1 = x0 + border, y0 + border, x1 - border, y1 - strip
            c.create_rectangle(ix0, iy0, ix1, iy1, fill="#2b2b2b" if images else "#f5f5f5", outline="", tags=(tag,))
            photo = self._photo(images[0], int(ix1 - ix0), int(iy1 - iy0)) if images else None
            if photo is not None:
                c.create_image((ix0 + ix1) / 2, (iy0 + iy1) / 2, image=photo, tags=(tag,))
            else:
                text = "loading..." if images and str(images[0]) in self._pending else (
                    "no thumbnail" if images else "no image")
                c.create_text((ix0 + ix1) / 2, (iy0 + iy1) / 2, fill="#aaa", font=("Segoe UI", small),
                              text=text, tags=(tag,))
            label = f"U{dev:02d}" + (f"   ({len(images)} images)" if len(images) > 1 else "")
            c.create_text(gx, y1 - strip / 2, text=label, fill=STRIP_TEXT if images else "#777",
                          font=("Segoe UI", strip_font, "bold"), tags=(tag,))
            c.tag_bind(tag, "<Button-1>", lambda e, d=dev: self.select(d))
            c.tag_bind(tag, "<Double-Button-1>", lambda e, d=dev: (self.select(d), self.open_device()))
            c.tag_bind(tag, "<Enter>", lambda e, d=dev: self._show_tip(e, d))
            c.tag_bind(tag, "<Leave>", lambda e: self._hide_tip())
        if not self._pending:
            ready()

    # ------------------------------------------------------------ thumbnails

    def _photo(self, path, width, height):
        """The image's thumbnail scaled to fit width x height, or None while it isn't there."""
        key = str(path)
        if key not in self._thumbs:
            if key not in self._pending:
                if not self._pending:
                    self.after(150, self._poll_thumbs)
                self._pending.add(key)
                self._todo.put((key, path, self._cache_dir(), self.thumb_px))
            return None
        base = self._thumbs[key]
        if base is None or width < 4 or height < 4:
            return None
        if self._photo_size != (width, height):
            self._photos, self._photo_size = {}, (width, height)
        if key not in self._photos:
            f = min(width / base.width, height / base.height)
            im = base.resize((max(1, round(base.width * f)), max(1, round(base.height * f))),
                             Image.Resampling.LANCZOS)
            self._photos[key] = ImageTk.PhotoImage(im)
        return self._photos[key]

    def _thumb_worker(self):
        while True:
            key, path, cache_dir, size = self._todo.get()
            try:
                im = make_thumb(path, cache_dir, size)
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
        if self._pending:
            done = sum(1 for v in self.devices.values() for p in v[:1] if str(p) in self._thumbs)
            status(f"Making thumbnails: {done} of {len(self.devices)} devices "
                   f"(the first time each image is read in full)...")
            self.after(150, self._poll_thumbs)
        if got:
            self.draw()

    # ------------------------------------------------------------ selection, tooltip

    def select(self, dev):
        self.selected = dev
        self.draw()
        self._show_selected()

    def _show_selected(self):
        images = self.devices.get(self.selected, []) if self.selected is not None else []
        if self.selected is None:
            self.sel_lbl.configure(text="Click a device on the map.\nDouble-click opens its image.")
        else:
            names = "\n".join(p.name for p in images) or "no image"
            self.sel_lbl.configure(text=f"Device {self.selected}\n{names}")
        self.open_btn.configure(state="normal" if images else "disabled")

    def _show_tip(self, e, dev):
        self._hide_tip()
        images = self.devices.get(dev, [])
        lines = [f"Device {dev}"] + ([p.name for p in images] or ["no image"])
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

    def open_device(self, dev=None):
        dev = self.selected if dev is None else dev
        if not self.devices.get(dev):
            return
        key = (str(self.folder), dev)
        win = self._windows.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
            return
        self._windows[key] = ImageWindow(self, dev)


class ImageWindow(tk.Toplevel):
    """The images of one device at full resolution (wheel zoom, drag to pan, double-click fits)."""

    def __init__(self, tab, dev):
        super().__init__(tab)
        self.tab = tab
        self.wafer = tab.wafer_var.get()
        self.devices = dict(tab.devices)   # stays with this wafer when the tab shows another one
        self._raw = None
        dpi = self.winfo_fpixels("1i") / 96
        w = min(int(1300 * dpi), int(self.winfo_screenwidth() * 0.9))
        h = min(int(900 * dpi), int(self.winfo_screenheight() * 0.85))
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth() - w) // 2}+{(self.winfo_screenheight() - h) // 3}")

        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="< Device", command=lambda: self._step_device(-1)).pack(side="left")
        ttk.Button(bar, text="Device >", command=lambda: self._step_device(1)).pack(side="left", padx=(2, 8))
        ttk.Label(bar, text="Image:").pack(side="left")
        self.image_box = ttk.Combobox(bar, state="readonly", width=40)
        self.image_box.pack(side="left", padx=4)
        self.image_box.bind("<<ComboboxSelected>>", lambda e: self._show(self.image_box.current()))
        ttk.Button(bar, text="Fit", command=lambda: self.view.fit()).pack(side="left", padx=4)
        self.hover = ttk.Label(bar, text="", foreground="#666", font=("Consolas", 9))
        self.hover.pack(side="right")
        self.info = ttk.Label(self, text="", foreground="#444", padding=(8, 2))
        self.info.pack(side="bottom", anchor="w", fill="x")
        self.view = ImageView(self, on_hover=self._on_hover)
        self.view.pack(fill="both", expand=True)
        self.after(50, lambda: self._set_device(dev))

    def _set_device(self, dev):
        self.dev = dev
        self.images = self.devices[dev]
        self.title(f"{self.wafer} - device {dev}")
        self.image_box.configure(values=[p.name for p in self.images])
        self._show(0)

    def _step_device(self, d):
        devs = sorted(self.devices)
        i = devs.index(self.dev)
        new = devs[(i + d) % len(devs)]
        # the map keeps its window per device: move this one to its new device
        self.tab._windows.pop((str(self.tab.folder), self.dev), None)
        self._set_device(new)
        if self.tab.wafer_var.get() == self.wafer:
            self.tab._windows[(str(self.tab.folder), new)] = self
            self.tab.select(new)

    def _show(self, i):
        path = self.images[i]
        try:
            with loading(f"Opening {path.name}..."):
                self._raw = io.load_image(path)
        except Exception as e:
            messagebox.showerror("Open image", f"Could not load {path}:\n{e}", parent=self)
            return
        self.image_box.current(i)
        self.view.set_image(self._raw)
        h, w = self._raw.shape[:2]
        self.info.configure(text=f"Device {self.dev}, image {i + 1} / {len(self.images)}   {path}   ({w} x {h} px)")

    def _on_hover(self, xy):
        if xy is None or self._raw is None:
            self.hover.configure(text="")
            return
        x, y = xy
        self.hover.configure(text=f"x {x:4d}  y {y:4d} px   RGB {tuple(int(v) for v in self._raw[y, x])}")
