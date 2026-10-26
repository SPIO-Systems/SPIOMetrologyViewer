"""The GUI: one tab per plate, inside it one tab per measurement type (from measurements.yaml), each
with Statistics, Wafer map and Step-by-step tuning (follow the pipeline on one image and tune the
parameters of each step).

    python -m spiometrologyviewer.gui [--measurements measurements.yaml]

Mouse wheel zooms the image, dragging pans it, double-click fits it to the window.
"""
import argparse
import copy
import math
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

if __name__ == "__main__" and not __package__:
    # Started as a plain script (e.g. PyCharm's Run on this file): make the package imports work.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "spiometrologyviewer"

from PIL import Image, ImageTk

from . import cli, gui_steps, io, measurements, pipeline, tuning, workspace
from .gui_steps import get_param, set_param

ROOT = workspace.ROOT
PANEL_WIDTH = 420  # at 96 dpi; scaled to the screen in App
WHEEL_EVENTS = ("<MouseWheel>", "<Button-4>", "<Button-5>")   # Windows / macOS, Linux (X11) up, down


def wheel_up(e):
    """True if a mouse wheel event scrolls up (away from the user), on any platform."""
    return e.num == 4 or getattr(e, "delta", 0) > 0


class ImageView(tk.Canvas):
    """Shows an RGB array with wheel zoom (about the cursor), drag to pan, double-click to fit."""

    def __init__(self, master, on_hover=None):
        super().__init__(master, background="#1e1e1e", highlightthickness=0)
        self.img = None
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self.auto_fit = True
        self.on_hover = on_hover
        self._photo = None
        self._drag = None
        self.marker = None   # (x, y, radius) in image px, drawn as a ring
        self.bind("<Configure>", lambda e: self.fit() if self.auto_fit else self.render())
        for ev in WHEEL_EVENTS:
            self.bind(ev, self._wheel)
        self.bind("<ButtonPress-1>", lambda e: setattr(self, "_drag", (e.x, e.y)))
        self.bind("<B1-Motion>", self._pan)
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<Motion>", self._motion)

    def set_image(self, rgb):
        new_size = self.img is None or self.img.size != (rgb.shape[1], rgb.shape[0])
        self.img = Image.fromarray(rgb)
        self.fit() if new_size or self.auto_fit else self.render()

    def fit(self):
        self.auto_fit = True
        if self.img is None:
            return
        cw, ch = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        w, h = self.img.size
        self.scale = min(cw / w, ch / h)
        self.ox, self.oy = (cw - w * self.scale) / 2, (ch - h * self.scale) / 2
        self.render()

    def render(self):
        self.delete("all")
        if self.img is None:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        w, h = self.img.size
        x0, y0 = max(0.0, -self.ox / self.scale), max(0.0, -self.oy / self.scale)
        x1, y1 = min(w, (cw - self.ox) / self.scale), min(h, (ch - self.oy) / self.scale)
        if x1 <= x0 or y1 <= y0:
            return
        size = (max(1, round((x1 - x0) * self.scale)), max(1, round((y1 - y0) * self.scale)))
        resample = Image.Resampling.NEAREST if self.scale >= 1 else Image.Resampling.BILINEAR
        self._photo = ImageTk.PhotoImage(self.img.resize(size, resample, box=(x0, y0, x1, y1)))
        self.create_image(self.ox + x0 * self.scale, self.oy + y0 * self.scale, image=self._photo, anchor="nw")
        if self.marker:
            mx, my, mr = self.marker
            cx, cy, r = self.ox + mx * self.scale, self.oy + my * self.scale, max(mr * self.scale, 8)
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#00ffff", width=2)

    def zoom_to(self, x, y, scale, marker_radius=None):
        """Centre image point (x, y) at `scale` display px per image px, optionally ringed."""
        self.auto_fit = False
        self.scale = scale
        self.ox = self.winfo_width() / 2 - x * scale
        self.oy = self.winfo_height() / 2 - y * scale
        self.marker = (x, y, marker_radius) if marker_radius else None
        self.render()

    def _wheel(self, e):
        new = min(max(self.scale * (1.25 if wheel_up(e) else 0.8), 0.05), 40)
        f = new / self.scale
        self.ox, self.oy = e.x - (e.x - self.ox) * f, e.y - (e.y - self.oy) * f
        self.scale = new
        self.auto_fit = False
        self.render()

    def _pan(self, e):
        if self._drag:
            self.ox += e.x - self._drag[0]
            self.oy += e.y - self._drag[1]
            self._drag = (e.x, e.y)
            self.auto_fit = False
            self.render()

    def _motion(self, e):
        if self.img is not None and self.on_hover:
            x, y = int((e.x - self.ox) / self.scale), int((e.y - self.oy) / self.scale)
            w, h = self.img.size
            self.on_hover((x, y) if 0 <= x < w and 0 <= y < h else None)


class ParamRow(ttk.Frame):
    """Label, slider and spin box for one parameter."""

    def __init__(self, master, spec, value, default, on_change, wrap):
        super().__init__(master)
        self.spec, self.on_change = spec, on_change
        self.decimals = 0 if spec.integer else max(0, -math.floor(math.log10(spec.step) + 1e-9))
        lo, hi = min(spec.lo, value, default), max(spec.hi, value, default)
        self.var = tk.IntVar(value=value) if spec.integer else tk.DoubleVar(value=value)

        ttk.Label(self, text=spec.label, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.default_lbl = ttk.Label(self, foreground="#888")
        self.default_lbl.grid(row=0, column=1, sticky="e")
        tk.Scale(self, from_=lo, to=hi, resolution=spec.step, orient="horizontal", showvalue=False,
                 variable=self.var, highlightthickness=0).grid(row=1, column=0, sticky="ew")
        spin_kw = {} if spec.integer else {"format": f"%.{self.decimals}f"}
        ttk.Spinbox(self, from_=lo, to=hi, increment=spec.step, textvariable=self.var, width=9,
                    **spin_kw).grid(row=1, column=1, padx=(6, 0))
        if spec.help:
            ttk.Label(self, text=spec.help, foreground="#666", wraplength=wrap,
                      font=("Segoe UI", 8)).grid(row=2, column=0, columnspan=2, sticky="w")
        self.columnconfigure(0, weight=1)
        self.default = default
        self._show_default(value)
        self.var.trace_add("write", self._changed)

    def _value(self):
        v = self.var.get()
        return int(round(v)) if self.spec.integer else round(float(v), self.decimals + 2)

    def _show_default(self, v):
        changed = abs(v - self.default) > 1e-9
        self.default_lbl.configure(text=f"default {self.default}" if changed else "",
                                   foreground="#c05000" if changed else "#888")

    def _changed(self, *_):
        try:
            v = self._value()
        except (tk.TclError, ValueError):
            return  # half-typed number
        self._show_default(v)
        self.on_change(self.spec, v)

    def set(self, v):
        self.var.set(v)


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame (`.inner`) that scrolls with the mouse wheel while hovered."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.bind("<Enter>", lambda e: [self.bind_all(ev, self._wheel) for ev in WHEEL_EVENTS])
        self.bind("<Leave>", lambda e: [self.unbind_all(ev) for ev in WHEEL_EVENTS])

    def _wheel(self, e):
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(-1 if wheel_up(e) else 1, "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.canvas.yview_moveto(0)


class App:
    """The step-by-step tuning tab. With a `measurement` (measurements.Measurement) it uses that
    type's parameters and images, and "Run on all images" runs its wafers into its output folder."""

    def __init__(self, root, config_path, image=None, parent=None, measurement=None):
        self.root = root
        self.container = parent or root   # where the widgets go (a notebook tab, or the window)
        self.on_batch_done = None         # called with the output folder after a successful batch run
        self.measurement = measurement
        self.config_path = Path(measurement.config if measurement else config_path)
        self.defaults = io.load_config(self.config_path)
        self.cfg = copy.deepcopy(self.defaults)
        # the region (mirror / lens) is not a tunable parameter, so the steps stay the same
        self.step_ui = gui_steps.steps_for(self.cfg)
        self.step_funcs = dict(pipeline.steps_for(self.cfg))
        self.tunings_dir = tuning.tunings_dir(self.config_path)   # timestamped copies of every tuning used
        self.step = 0
        self.reached = 0
        self.base = None        # state before the first step (the loaded image)
        self.states = []        # states[i] = state after step i; only a valid prefix is kept
        self.gen = 0            # bumped on every change; results of older computations are dropped
        self.pending = None
        self.images = []
        self.view_choice = {}
        self.rows = []
        self.results = queue.Queue()
        self.batch_running = False

        self.pw = int(PANEL_WIDTH * root.winfo_fpixels("1i") / 96)
        self._build()
        self._show_tuning()
        self._show_step()
        root.after(50, self._poll)

        if image is None and measurement is not None:
            image = measurement.first_image(self.cfg)
        if image is not None:
            self.open_image(Path(image))

    # ------------------------------------------------------------ layout

    def _build(self):
        bar = ttk.Frame(self.container, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Open image...", command=self._ask_image).pack(side="left")
        ttk.Button(bar, text="<", width=3, command=lambda: self._step_image(-1)).pack(side="left", padx=(8, 0))
        self.image_box = ttk.Combobox(bar, state="readonly", width=34)
        self.image_box.pack(side="left")
        self.image_box.bind("<<ComboboxSelected>>", lambda e: self.open_image(self.images[self.image_box.current()]))
        ttk.Button(bar, text=">", width=3, command=lambda: self._step_image(1)).pack(side="left")
        ttk.Button(bar, text="Fit image", command=lambda: self.view.fit()).pack(side="left", padx=8)
        self.tuning_lbl = ttk.Label(bar, text="", font=("Segoe UI", 9, "bold"))
        self.tuning_lbl.pack(side="left", padx=(8, 0))
        self.status = ttk.Label(bar, text="", foreground="#444")
        self.status.pack(side="left", padx=8)
        self.hover = ttk.Label(bar, text="", foreground="#666", font=("Consolas", 9))
        self.hover.pack(side="right")

        body = ttk.PanedWindow(self.container, orient="horizontal")
        body.pack(fill="both", expand=True)
        self.view = ImageView(body, on_hover=self._on_hover)
        body.add(self.view, weight=1)

        panel = ttk.Frame(body, padding=8, width=self.pw)
        body.add(panel, weight=0)

        self.crumbs = ttk.Frame(panel)
        self.crumbs.pack(fill="x")
        self.title = ttk.Label(panel, font=("Segoe UI", 14, "bold"))
        self.title.pack(anchor="w", pady=(8, 2))
        self.text = ttk.Label(panel, wraplength=self.pw - 20, justify="left")
        self.text.pack(anchor="w", fill="x")

        views = ttk.Frame(panel)
        views.pack(fill="x", pady=(6, 0))
        ttk.Label(views, text="Show:").pack(side="left")
        self.view_var = tk.StringVar()
        self.view_box = ttk.Combobox(views, textvariable=self.view_var, state="readonly", width=32)
        self.view_box.pack(side="left", padx=4)
        self.view_box.bind("<<ComboboxSelected>>", lambda e: self._view_changed())

        # Bottom first so it keeps its space when the window gets small.
        nav = ttk.Frame(panel)
        nav.pack(side="bottom", fill="x", pady=(6, 0))
        self.prev_btn = ttk.Button(nav, text="< Previous step", command=lambda: self._go(self.step - 1))
        self.prev_btn.pack(side="left")
        self.next_btn = ttk.Button(nav, text="Next step >", command=lambda: self._go(self.step + 1))
        self.next_btn.pack(side="right")

        self.finish = ttk.LabelFrame(panel, text="Use these parameters", padding=6)
        ttk.Button(self.finish, text="Save parameters...", command=self._save).pack(side="left")
        self.run_btn = ttk.Button(self.finish, text="Run on all images...", command=self._run_all)
        self.run_btn.pack(side="left", padx=6)

        self.info = tk.Text(panel, width=40, height=12, font=("Consolas", 9), wrap="word", relief="flat",
                            background="#f4f4f4", state="disabled")
        self.info.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Label(panel, text="This image", font=("Segoe UI", 9, "bold")).pack(side="bottom", anchor="w")

        ttk.Separator(panel).pack(fill="x", pady=6)
        head = ttk.Frame(panel)
        head.pack(fill="x")
        ttk.Label(head, text="Parameters", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Reset step to defaults", command=self._reset_step).pack(side="right")
        self.params = ScrollFrame(panel)
        self.params.pack(fill="both", expand=True, pady=(4, 0))

    def _show_step(self):
        ui = self.step_ui[self.step]
        for w in self.crumbs.winfo_children():
            w.destroy()
        for i, s in enumerate(self.step_ui):
            color = "#0060c0" if i == self.step else ("#333" if i <= self.reached else "#aaa")
            lbl = ttk.Label(self.crumbs, text=f"{i + 1} {s.key}", foreground=color, cursor="hand2",
                            font=("Segoe UI", 8, "bold" if i == self.step else "normal"))
            lbl.pack(side="left", padx=(0, 6))
            lbl.bind("<Button-1>", lambda e, i=i: self._go(i) if i <= self.reached else None)

        self.title.configure(text=f"Step {self.step + 1} / {len(self.step_ui)}: {ui.title}")
        self.text.configure(text=ui.text)
        names = list(ui.views)
        self.view_box.configure(values=names)
        self.view_var.set(self.view_choice.get(self.step, names[0]))

        self.params.clear()
        self.rows = []
        for spec in ui.params:
            row = ParamRow(self.params.inner, spec, get_param(self.cfg, spec.path),
                           get_param(self.defaults, spec.path), self._param_changed, self.pw - 50)
            row.pack(fill="x", pady=(0, 8), padx=(0, 6))
            self.rows.append(row)

        last = self.step == len(self.step_ui) - 1
        self.prev_btn.configure(state="normal" if self.step > 0 else "disabled")
        self.next_btn.configure(state="disabled" if last else "normal")
        if last:
            self.finish.pack(side="bottom", fill="x", pady=(6, 0), before=self.info)
        else:
            self.finish.pack_forget()

    # ------------------------------------------------------------ actions

    def _ask_image(self):
        start = self.images[0].parent if self.images else ROOT
        p = filedialog.askopenfilename(initialdir=start, filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
        if p:
            self.open_image(Path(p))

    def _step_image(self, d):
        if self.images:
            i = (self.image_box.current() + d) % len(self.images)
            self.open_image(self.images[i])

    def open_image(self, path):
        try:
            self.base = pipeline.initial_state(path)
        except Exception as e:
            messagebox.showerror("Open image", f"Could not load {path}:\n{e}")
            return
        self.images = io.list_images(path.parent)
        self.image_box.configure(values=[p.name for p in self.images])
        self.image_box.current([p.resolve() for p in self.images].index(path.resolve()))
        self._invalidate(0)
        self._request(0)

    def _go(self, step):
        if 0 <= step < len(self.step_ui):
            self.step = step
            self.reached = max(self.reached, step)
            self._show_step()
            self._request(0)

    def _param_changed(self, spec, value):
        if get_param(self.cfg, spec.path) == value:
            return
        set_param(self.cfg, spec.path, value)
        self._invalidate(self.step)
        self._request()
        self._show_tuning()

    def _show_tuning(self):
        """Which tuning the current parameters are: an existing one, or new (made on Save / Run)."""
        name = tuning.find(self.cfg, self.tunings_dir)
        if name:
            self.tuning_lbl.configure(text=f"Tuning: {name}", foreground="#2e7d32")
        else:
            self.tuning_lbl.configure(text="Tuning: new (named when you save or run it)", foreground="#c05000")

    def _register_tuning(self, cfg):
        name = tuning.register(cfg, self.tunings_dir, template=self.config_path)
        self._show_tuning()
        return name

    def _reset_step(self):
        for row in self.rows:
            row.set(get_param(self.defaults, row.spec.path))

    def _view_changed(self):
        self.view_choice[self.step] = self.view_var.get()
        self._render()

    def _save(self):
        p = filedialog.asksaveasfilename(initialdir=self.config_path.parent, initialfile=self.config_path.name,
                                         defaultextension=".yaml", filetypes=[("YAML", "*.yaml")])
        if p:
            io.save_config(p, self.cfg, template=self.config_path)
            name = self._register_tuning(self.cfg)
            self.status.configure(text=f"Saved parameters to {p}  (tuning {name})")

    def _run_all(self):
        if self.batch_running:
            return
        if self.measurement is not None:
            self._run_measurement()
            return
        if not self.images:
            return
        folder = filedialog.askdirectory(
            title="Images to analyse: one wafer folder, or a folder of wafer folders (e.g. Images/P2)",
            initialdir=self.images[0].parent.parent)
        if not folder:
            return
        out = filedialog.askdirectory(title="Output folder for the results (one subfolder per wafer)",
                                      initialdir=ROOT, mustexist=False)
        if not out:
            return
        cfg = copy.deepcopy(self.cfg)
        name = self._register_tuning(cfg)
        self._start_batch(lambda progress: cli.run_wafers(folder, out, cfg, progress=progress, tuning_name=name,
                                                          template=self.config_path), out)

    def _run_measurement(self):
        """Every wafer of this measurement type, with the current parameters, into its output folder."""
        m = self.measurement
        busy = workspace.lock_info(m.output)
        if busy and not messagebox.askokcancel(
                "Run on all images", f"{m.key} is being run already:\n{busy}\n\nRunning it at the same time "
                "overwrites each other's results. Only go on if that run has stopped."):
            return
        cfg = copy.deepcopy(self.cfg)
        wafers = m.wafers(cfg)
        if not wafers:
            messagebox.showerror("Run on all images", f"No wafer folders found in {m.images}")
            return
        done = [w for w in wafers if (m.output / w / "results.csv").exists()]
        current = tuning.find(cfg, self.tunings_dir) or "a new tuning"
        text = (f"Analyse the {len(wafers)} wafers of {m.key} with the current parameters ({current})?\n"
                f"Results go to {m.output}")
        skip = False
        if done:
            import pandas as pd
            old = set()
            for w in done:
                try:
                    old.add(tuning.of_results(m.output / w, pd.read_csv(m.output / w / "results.csv"), self.tunings_dir))
                except Exception:
                    old.add(tuning.UNKNOWN)
            other = sorted(old - {current})
            warn = (f"\nThey were made with {', '.join(other)}: with No, the wafers will have different "
                    "tunings and can't be compared directly." if other else "")
            answer = messagebox.askyesnocancel(
                "Run on all images", text + f"\n\n{len(done)} of them already have results. Run those again?\n"
                "Yes = run all wafers, No = only the wafers without results." + warn)
            if answer is None:
                return
            skip = not answer
        elif not messagebox.askokcancel("Run on all images", text):
            return
        name = self._register_tuning(cfg)
        def run(progress):
            with workspace.Lock(m.output):
                cli.run_wafers(m.images, m.output, cfg, skip_done=skip, progress=progress, wafers=wafers,
                               tuning_name=name, template=self.config_path)

        self._start_batch(run, m.output)

    def _start_batch(self, run, out):
        self.batch_running = True
        self.run_btn.configure(state="disabled")

        def work():
            try:
                run(lambda msg: self.results.put(("progress", msg)))
                self.results.put(("batch_done", f"Done. Results in {out}", out))
            except Exception:
                self.results.put(("batch_done", "Batch run failed:\n" + traceback.format_exc(), None))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ computation

    def _invalidate(self, step):
        del self.states[step:]
        self.gen += 1

    def _request(self, delay=250):
        """Recompute up to the current step after `delay` ms (debounces slider drags)."""
        if self.pending:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(delay, self._start)

    def _start(self):
        self.pending = None
        if self.base is None:
            return
        if len(self.states) > self.step:
            self._render()
            return
        self.gen += 1
        gen, start, target = self.gen, len(self.states), self.step
        state = self.states[-1] if self.states else self.base
        cfg = copy.deepcopy(self.cfg)
        self.status.configure(text=f"Computing {self.step_ui[start].key} ... {self.step_ui[target].key}", foreground="#c05000")

        def work():
            t0, new = time.perf_counter(), []
            try:
                s = state
                for ui in self.step_ui[start:target + 1]:
                    if gen != self.gen:
                        return
                    s = {**s, **self.step_funcs[ui.key](s, cfg)}
                    new.append(s)
                self.results.put(("states", gen, start, new, None, time.perf_counter() - t0))
            except Exception:
                self.results.put(("states", gen, start, new, traceback.format_exc(), 0))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                msg = self.results.get_nowait()
                if msg[0] == "states":
                    self._computed(*msg[1:])
                elif msg[0] == "progress":
                    self.status.configure(text=msg[1], foreground="#444")
                elif msg[0] == "batch_done":
                    self.batch_running = False
                    self.run_btn.configure(state="normal")
                    self.status.configure(text=msg[1].splitlines()[0], foreground="#444")
                    (messagebox.showerror if "failed" in msg[1] else messagebox.showinfo)("Run on all images", msg[1])
                    if msg[2] is not None and self.on_batch_done:
                        self.on_batch_done(Path(msg[2]))
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _computed(self, gen, start, new, error, seconds):
        if gen != self.gen or len(self.states) != start:
            return  # parameters changed meanwhile
        self.states.extend(new)
        if error:
            self.status.configure(text="Error, see 'This image' box", foreground="#c00000")
            self._set_info(error)
            return
        self.status.configure(text=f"{self.base['name']}  ({seconds:.1f} s)", foreground="#444")
        self._render()

    def _render(self):
        if len(self.states) <= self.step:
            return
        ui, s = self.step_ui[self.step], self.states[self.step]
        try:
            self.view.set_image(ui.views[self.view_var.get()](s, self.cfg))
            self._set_info(ui.info(s, self.cfg) if ui.info else "")
        except Exception:
            self._set_info(traceback.format_exc())

    def _set_info(self, text):
        self.info.configure(state="normal", height=min(12, max(4, text.count("\n") + 2)))
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")

    def _on_hover(self, xy):
        if xy is None or self.base is None:
            self.hover.configure(text="")
            return
        x, y = xy
        px = self.cfg["um_per_px"]
        text = f"x {x:4d}  y {y:4d} px  ({x * px:6.0f}, {y * px:6.0f} um)   RGB {tuple(int(v) for v in self.base['raw'][y, x])}"
        s = self.states[-1] if self.states else {}
        if "ratio" in s:
            text += f"   flattened {s['ratio'][y, x]:.2f}"
        self.hover.configure(text=text)


class MeasurementTab(ttk.Frame):
    """One measurement type (e.g. P2 / With mirror): Statistics, Wafer map and Step-by-step tuning,
    all on that type's images, parameters, tolerances and results.

    The contents are built the first time the tab is shown, the tuning part only when it is opened
    (it loads an image and starts computing)."""

    def __init__(self, master, root, measurement):
        super().__init__(master)
        self.root, self.m = root, measurement
        self.app = self.wafer_tab = self.stats = None
        self.bind("<Map>", lambda e: self.ensure_built())

    def ensure_built(self):
        if self.wafer_tab is not None:
            return
        from .gui_stats import StatsPanel
        from .gui_wafer import WaferMapTab

        try:
            from . import analyses
            analysis = analyses.name_of(self.m.load_config())
        except OSError as e:
            ttk.Label(self, text=f"Can't read {self.m.config}:\n{e}", foreground="#c00000").pack(pady=20)
            self.wafer_tab = False
            return
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="  Statistics  ")
        self.wafer_tab = WaferMapTab(self.notebook, self.m, analysis)
        self.notebook.add(self.wafer_tab, text="  Wafer map  ")
        self.tuning = ttk.Frame(self.notebook)
        self.notebook.add(self.tuning, text="  Step-by-step tuning  ")
        self.stats = StatsPanel(stats_frame, self.wafer_tab)
        self.stats.pack(fill="both", expand=True)

        self.wafer_tab.open_in_tuning = self.open_in_tuning
        self.wafer_tab.on_show = lambda: self.notebook.select(self.wafer_tab)
        self.tuning.bind("<Map>", lambda e: self.ensure_app())

    def ensure_app(self, image=None):
        if self.app is None:
            self.app = App(self.root, self.m.config, image, parent=self.tuning, measurement=self.m)
            self.app.on_batch_done = lambda out: self.wafer_tab.set_results_root(self.m.output)
            return True
        return False

    def open_in_tuning(self, path):
        self.notebook.select(self.tuning)
        if not self.ensure_app(Path(path)):
            self.app.open_image(Path(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measurements", type=Path, default=measurements.DEFAULT_REGISTRY,
                    help="plates and measurement types (default measurements.yaml)")
    args = ap.parse_args(argv)
    try:  # sharp rendering on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    if not args.measurements.exists() and choose_workspace(root):
        root.destroy()   # the workspace is read at import: start again with the new one
        subprocess.Popen([sys.executable, "-m", "spiometrologyviewer.gui"], cwd=workspace.REPO)
        return
    build(root, args.measurements)
    root.mainloop()


def choose_workspace(root):
    """First start on a PC (no measurements.yaml found): ask for the shared workspace folder, e.g.
    the 24 SPIOMetrologyViewer folder in Nextcloud, and remember it in settings.yaml. An empty folder is set up
    from workspace_template. True if a workspace was chosen."""
    root.withdraw()
    messagebox.showinfo("SPIOMetrologyViewer", "Choose the shared SPIOMetrologyViewer workspace folder (e.g. the "
                        "24 SPIOMetrologyViewer folder in your Nextcloud). It holds measurements.yaml, the configs, "
                        "images and results.", parent=root)
    folder = filedialog.askdirectory(title="SPIOMetrologyViewer workspace folder", parent=root)
    if not folder:
        return False
    if not workspace.is_workspace(folder):
        if any(Path(folder).iterdir()):
            messagebox.showerror("SPIOMetrologyViewer", f"{folder} has no measurements.yaml: that's not the "
                                 "workspace folder.", parent=root)
            return False
        if not messagebox.askyesno("SPIOMetrologyViewer", f"{folder} is empty. Set up a new workspace in it?",
                                   parent=root):
            return False
        workspace.create(folder)
    workspace.save_settings(folder)
    return True


def build(root, registry=measurements.DEFAULT_REGISTRY):
    """The window: one tab per plate, inside it one tab per measurement type (MeasurementTab).
    Returns {"P2/With mirror": MeasurementTab, ...}."""
    root.title("SPIOMetrologyViewer")
    root.geometry("1450x950")
    try:
        root.state("zoomed")                  # Windows
    except tk.TclError:
        try:
            root.attributes("-zoomed", True)  # Linux
        except tk.TclError:
            pass
    style = ttk.Style(root)
    style.configure("Plate.TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=(14, 4))
    style.configure("Type.TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(10, 3))

    try:
        plates = measurements.load(registry)
    except Exception as e:
        ttk.Label(root, text=f"Can't read {registry}:\n{e}", foreground="#c00000").pack(pady=40)
        return {}
    tabs = {}
    notebook = ttk.Notebook(root, style="Plate.TNotebook")
    notebook.pack(fill="both", expand=True)
    for plate, types in plates.items():
        plate_nb = ttk.Notebook(notebook, style="Type.TNotebook")
        notebook.add(plate_nb, text=f"  {plate}  ")
        for m in types:
            tab = MeasurementTab(plate_nb, root, m)
            plate_nb.add(tab, text=f"  {m.name}  ")
            tabs[m.key] = tab
    if not tabs:
        ttk.Label(root, text=f"No measurement types in {registry}", foreground="#c00000").pack(pady=40)
    return tabs


if __name__ == "__main__":
    main()
