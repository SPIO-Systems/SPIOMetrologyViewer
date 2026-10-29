"""Browse tab: the images of a measurement type without an analysis (config `analysis: none`, e.g.
P1 and P5 until they are implemented). Pick a wafer and step through its images; no pipeline,
tolerances or batch run."""
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from . import io
from .gui import ImageView, loading


class BrowseTab(ttk.Frame):
    def __init__(self, master, measurement):
        super().__init__(master)
        self.m = measurement
        self.cfg = measurement.load_config()
        self.wafers = {}   # label -> folder
        self.images = []

        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Wafer:").pack(side="left")
        self.wafer_var = tk.StringVar()
        self.wafer_box = ttk.Combobox(bar, textvariable=self.wafer_var, state="readonly", width=30)
        self.wafer_box.pack(side="left", padx=(2, 0))
        self.wafer_box.bind("<<ComboboxSelected>>", lambda e: self._wafer_chosen())
        ttk.Button(bar, text="<", width=3, command=lambda: self._step(-1)).pack(side="left", padx=(8, 0))
        self.image_box = ttk.Combobox(bar, state="readonly", width=40)
        self.image_box.pack(side="left")
        self.image_box.bind("<<ComboboxSelected>>", lambda e: self._show(self.image_box.current()))
        ttk.Button(bar, text=">", width=3, command=lambda: self._step(1)).pack(side="left")
        ttk.Button(bar, text="Fit image", command=lambda: self.view.fit()).pack(side="left", padx=8)
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        self.hover = ttk.Label(bar, text="", foreground="#666", font=("Consolas", 9))
        self.hover.pack(side="right")
        ttk.Label(self, text=f"{measurement.key}: no analysis yet, the images can only be browsed. "
                             f"Images from {measurement.base / measurement.images}", foreground="#666",
                  padding=(8, 0)).pack(side="top", anchor="w")
        self.info = ttk.Label(self, text="", foreground="#444", padding=(8, 2))
        self.info.pack(side="bottom", anchor="w", fill="x")
        self.view = ImageView(self, on_hover=self._on_hover)
        self.view.pack(fill="both", expand=True)
        self._raw = None
        self.refresh()

    def refresh(self):
        with loading(f"Looking for wafers of {self.m.key}..."):
            self.wafers = self.m.browsable(self.cfg)
        self.wafer_box.configure(values=list(self.wafers))
        if not self.wafers:
            self.info.configure(text=f"No wafer folders in {self.m.base / self.m.images}")
            return
        if self.wafer_var.get() not in self.wafers:
            self.wafer_var.set(next(iter(self.wafers)))
        self._wafer_chosen()

    def _wafer_chosen(self):
        folder = self.wafers.get(self.wafer_var.get())
        if folder is None:
            return
        with loading(f"Looking for images in {folder.name}..."):
            self.images = self.m.images_of(folder, self.cfg)
        self.image_box.configure(values=[self._label(p, folder) for p in self.images])
        if self.images:
            self._show(0)
        else:
            self.image_box.set("")
            self.info.configure(text=f"No images in {folder}")

    @staticmethod
    def _label(path, folder):
        return Path(path).relative_to(folder).as_posix()

    def _step(self, d):
        if self.images:
            self._show((self.image_box.current() + d) % len(self.images))

    def _show(self, i):
        path = self.images[i]
        try:
            with loading(f"Opening {path.name}..."):
                self._raw = io.load_image(path)
        except Exception as e:
            messagebox.showerror("Open image", f"Could not load {path}:\n{e}")
            return
        self.image_box.current(i)
        self.view.set_image(self._raw)
        h, w = self._raw.shape[:2]
        self.info.configure(text=f"{i + 1} / {len(self.images)}   {path}   ({w} x {h} px)")

    def _on_hover(self, xy):
        if xy is None or self._raw is None:
            self.hover.configure(text="")
            return
        x, y = xy
        self.hover.configure(text=f"x {x:4d}  y {y:4d} px   RGB {tuple(int(v) for v in self._raw[y, x])}")
