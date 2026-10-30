"""Wafer statistics tab: one parameter (or two) plotted over the wafers of one measurement type.

Views: yield bars (pass / warn / fail per wafer), distribution curves (cumulative, per wafer), the
average wafer map (per position over the selected wafers) and parameter vs parameter."""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm, colors as mcolors                                      # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure                                               # noqa: E402
from matplotlib.patches import Rectangle                                           # noqa: E402
import numpy as np                                                                 # noqa: E402

from . import wafer                                                                # noqa: E402
from .gui import ScrollFrame, loading, status                                      # noqa: E402
from .gui_wafer import COLORS, NAMES, _pitch, fmt                                  # noqa: E402

VIEWS = ["Yield bars", "Distribution curves", "Average wafer map", "Parameter vs parameter"]
MAP_STATS = ["mean", "median", "max", "std", "% fail"]
STATUS_ORDER = ["pass", "warn", "fail", "nodata"]
SUMMARY_COLUMNS = [("wafer", "Wafer", 190), ("tuning", "Tuning", 200), ("n", "n", 40), ("mean", "Mean", 70), ("median", "Median", 70),
                   ("std", "Std", 70), ("min", "Min", 70), ("max", "Max", 70), ("pass_pct", "% pass", 60),
                   ("warn_pct", "% warn", 60), ("fail_pct", "% fail", 60), ("nodata_pct", "% no data", 70)]
TEXT_COLUMNS = ("wafer", "tuning")


def _short_names(names):
    """Wafer names without their common prefix (17013_2_P2_05 -> 05), for plot labels."""
    long = [n for n in names if "_" in n]
    prefix = os.path.commonprefix(long) if len(long) > 1 else ""
    prefix = prefix[:prefix.rfind("_") + 1]
    return {n: n[len(prefix):] if prefix and n.startswith(prefix) and len(n) > len(prefix) else n
            for n in names}, prefix


class StatsPanel(ttk.Frame):
    """Uses the wafers, positions and tolerances of `tab` (the same type's WaferMapTab), so
    tolerance changes there redraw it."""

    def __init__(self, master, tab):
        super().__init__(master)
        self.tab = tab
        self.pw = int(320 * self.winfo_fpixels("1i") / 96)

        self.df = None
        self.checks = {}      # wafer name -> BooleanVar
        self.colors = {}      # wafer name -> plot colour
        # a value to look at first (P4: fitted R) has no pass / fail of its own, so its view is the curves
        lead = next((m.label for m in tab.metrics if m.follows), None)
        self.view = tk.StringVar(value=VIEWS[1] if lead else VIEWS[0])
        self.x_var = tk.StringVar(value=lead or wafer.OVERALL)
        self.y_var = tk.StringVar()
        self.map_stat = tk.StringVar(value="mean")
        self.log_x = tk.BooleanVar(value=False)
        self.show_tol = tk.BooleanVar(value=True)
        self.sort_worst = tk.BooleanVar(value=True)
        self.summary = None
        self.tunings = {}     # wafer name -> tuning name(s)
        self._picked = {}     # scatter artist -> rows
        self._map_cells = []  # (x0, y0, x1, y1, text) for the hover text on the average map

        self._build()
        tab.stats_window = self
        self.reload()

    # ------------------------------------------------------------ layout

    def _build(self):
        left = ttk.Frame(self, padding=8, width=self.pw)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y")

        view = ttk.LabelFrame(left, text="View", padding=6)
        view.pack(fill="x")
        for v in VIEWS:
            ttk.Radiobutton(view, text=v, value=v, variable=self.view, command=self._view_changed).pack(anchor="w")

        par = ttk.LabelFrame(left, text="Parameter", padding=6)
        par.pack(fill="x", pady=(8, 0))
        self.x_lbl = ttk.Label(par, text="Parameter")
        self.x_lbl.pack(anchor="w")
        self.x_box = ttk.Combobox(par, textvariable=self.x_var, state="readonly")
        self.x_box.pack(fill="x")
        self.x_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        self.y_lbl = ttk.Label(par, text="Y parameter")
        self.y_box = ttk.Combobox(par, textvariable=self.y_var, state="readonly")
        self.y_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        self.opts = ttk.LabelFrame(left, text="Options", padding=6)
        self.opts.pack(fill="x", pady=(8, 0))
        self.opt_widgets = {
            "stat": ttk.Frame(self.opts),
            "log": ttk.Checkbutton(self.opts, text="Log x axis", variable=self.log_x, command=self.redraw),
            "sort": ttk.Checkbutton(self.opts, text="Worst wafer on top", variable=self.sort_worst,
                                    command=self.redraw),
            "tol": ttk.Checkbutton(self.opts, text="Show pass / fail cutoffs", variable=self.show_tol,
                                   command=self.redraw),
        }
        ttk.Label(self.opt_widgets["stat"], text="Per position:").pack(side="left")
        box = ttk.Combobox(self.opt_widgets["stat"], textvariable=self.map_stat, values=MAP_STATS,
                           state="readonly", width=8)
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        wf = ttk.LabelFrame(left, text="Wafers", padding=6)
        wf.pack(fill="both", expand=True, pady=(8, 0))
        self.tuning_warn = ttk.Label(wf, foreground="#c62828", font=("Segoe UI", 9, "bold"),
                                     wraplength=self.pw - 40, justify="left")
        self.tuning_warn.pack(side="bottom", fill="x", pady=(4, 0))
        btns = ttk.Frame(wf)
        btns.pack(fill="x")
        for text, fn in (("All", lambda: self._set_all(True)), ("None", lambda: self._set_all(False)),
                         ("Invert", self._invert)):
            ttk.Button(btns, text=text, width=7, command=fn).pack(side="left", padx=(0, 4))
        self.wafer_list = ScrollFrame(wf)
        self.wafer_list.pack(fill="both", expand=True, pady=(4, 0))

        right = ttk.PanedWindow(self, orient="vertical")
        right.pack(side="left", fill="both", expand=True)
        plot = ttk.Frame(right)
        right.add(plot, weight=3)
        self.fig = Figure(dpi=self.winfo_fpixels("1i") * 0.75)   # a bit smaller than 1:1 so more fits
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        bar = ttk.Frame(plot)
        bar.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(self.canvas, bar, pack_toolbar=False).pack(side="left")
        self.status = ttk.Label(bar, foreground="#444")
        self.status.pack(side="left", padx=8)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        table = ttk.Frame(right, padding=(6, 4))
        right.add(table, weight=1)
        head = ttk.Frame(table)
        head.pack(fill="x")
        self.table_lbl = ttk.Label(head, font=("Segoe UI", 9, "bold"))
        self.table_lbl.pack(side="left")
        ttk.Button(head, text="Refresh", command=self.reload).pack(side="right")
        ttk.Button(head, text="Export table CSV...", command=self._export).pack(side="right", padx=4)
        cols = [c for c, _, _ in SUMMARY_COLUMNS]
        dpi = self.winfo_fpixels("1i") / 96
        self.tree = ttk.Treeview(table, columns=cols, show="headings", height=6)
        for c, text, width in SUMMARY_COLUMNS:
            self.tree.heading(c, text=text)
            self.tree.column(c, width=int(width * dpi), anchor="w" if c in TEXT_COLUMNS else "e",
                             stretch=c == "wafer")
        sb = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, pady=(4, 0))
        sb.pack(side="right", fill="y", pady=(4, 0))
        self._view_changed(draw=False)

    def _view_changed(self, draw=True):
        v = self.view.get()
        scatter = v == "Parameter vs parameter"
        self.x_lbl.configure(text="X parameter" if scatter else "Parameter")
        if scatter:
            self.y_lbl.pack(anchor="w", pady=(6, 0))
            self.y_box.pack(fill="x")
        else:
            self.y_lbl.pack_forget()
            self.y_box.pack_forget()
        shown = {"Yield bars": ["sort"], "Distribution curves": ["log", "tol"],
                 "Average wafer map": ["stat"], "Parameter vs parameter": ["tol"]}[v]
        for k, wdg in self.opt_widgets.items():
            if k in shown:
                wdg.pack(anchor="w", pady=1)
            else:
                wdg.pack_forget()
        self._fill_params()
        if draw:
            self.redraw()

    # ------------------------------------------------------------ data

    def reload(self):
        """Read every wafer the map tab knows about (Refresh, or a new results folder)."""
        try:
            with loading("Statistics: reading the results..."):
                self.df = wafer.load_all(self.tab.wafers, self.tab.positions, self.tab.tunings_dir,
                                         progress=lambda i, n, name: status(f"Statistics: reading {name} ({i} of {n})..."))
        except Exception as e:
            messagebox.showerror("Wafer statistics", f"Could not load the results:\n{e}", parent=self)
            self.df = None
        names = list(self.tab.wafers)
        self.tunings = ({} if self.df is None or self.df.empty else
                        {n: wafer.tuning_of(g) for n, g in self.df.groupby("wafer", sort=False)})
        palette = [mcolors.to_hex(c) for c in cm.tab10.colors] + [mcolors.to_hex(c) for c in cm.Dark2.colors]
        self.colors = {n: palette[i % len(palette)] for i, n in enumerate(names)}
        old = {n: v.get() for n, v in self.checks.items()}
        self.wafer_list.clear()
        self.checks = {}
        for n in names:
            var = tk.BooleanVar(value=old.get(n, "(top level)" not in n))   # top level: older copy of a wafer
            self.checks[n] = var
            row = ttk.Frame(self.wafer_list.inner)
            row.pack(fill="x", pady=1)
            tk.Label(row, background=self.colors[n], width=2).pack(side="left", padx=(0, 4))
            ttk.Checkbutton(row, text=n, variable=var, command=self.redraw).pack(side="left")
            ttk.Label(self.wafer_list.inner, text=f"tuning: {self.tunings.get(n, wafer.tuning.UNKNOWN)}",
                      foreground="#777", font=("Segoe UI", 8)).pack(anchor="w", padx=(36, 0))
        self._fill_params()
        self.redraw()

    def _fill_params(self):
        if self.df is None or self.df.empty:
            return
        v = self.view.get()
        params = wafer.parameters(self.df, self.tab.metrics, with_overall=True)
        if v == "Yield bars":   # needs pass / warn / fail
            params = [p for p in params if p.column is None or p.metric is not None]
        else:                   # needs values
            params = [p for p in params if p.column is not None]
        self.params = {p.label: p for p in params}
        labels = list(self.params)
        self.x_box.configure(values=labels)
        self.y_box.configure(values=labels)
        if self.x_var.get() not in self.params:
            self.x_var.set(labels[0])
        if self.y_var.get() not in self.params:
            offs = [l for l in labels if l.startswith("offset_y_um")]
            self.y_var.set(offs[0] if offs else labels[min(1, len(labels) - 1)])

    def selected(self):
        return [n for n, v in self.checks.items() if v.get()]

    def _set_all(self, on):
        for v in self.checks.values():
            v.set(on)
        self.redraw()

    def _invert(self):
        for v in self.checks.values():
            v.set(not v.get())
        self.redraw()

    # ------------------------------------------------------------ drawing

    def redraw(self):
        if not self.winfo_exists():
            return
        self._fill_params()
        self.fig.clear()
        self._picked, self._map_cells = {}, []
        self.status.configure(text="")
        wafers = self.selected()
        ax = self.fig.add_subplot(111)
        param = self.params.get(self.x_var.get()) if self.df is not None and not self.df.empty else None
        if param is None or not wafers:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No results loaded." if param is None else "Tick at least one wafer.",
                    ha="center", va="center", color="#888", transform=ax.transAxes)
            self.summary = None
            self.tuning_warn.configure(text="")
            self._fill_table(None)
            self.canvas.draw_idle()
            return
        df = self.df[self.df["wafer"].isin(wafers)]
        self.short, prefix = _short_names(wafers)
        view = self.view.get()
        if view == "Yield bars":
            self._yield(ax, df, wafers, param)
        elif view == "Distribution curves":
            self._curves(ax, df, wafers, param)
        elif view == "Average wafer map":
            self._wafer_map(ax, df, wafers, param)
        else:
            self._scatter(ax, df, wafers, param, self.params.get(self.y_var.get()))
        if prefix and view != "Average wafer map":
            self.fig.text(0.995, 0.995, f"wafer names without '{prefix}'", ha="right", va="top",
                          fontsize=7, color="#888")
        self._tuning_note(df)
        self.summary = wafer.summary(df, param, self.tab.metrics, order=wafers)
        self._fill_table(param)
        try:
            self.fig.tight_layout(rect=(0, 0.035, 1, 1))   # room for the tuning note
        except Exception:
            pass
        self.canvas.draw_idle()

    def _tuning_note(self, df):
        """Which tuning the plotted data comes from, at the bottom of the figure (so it is in a
        saved image too); a warning there and in the left panel if the ticked wafers differ."""
        names = sorted({n for t in df.groupby("wafer")["tuning"].unique() for n in t if isinstance(n, str)})
        if len(names) > 1:
            per = df.groupby("wafer")["tuning"].first()
            counts = {n: int((per == n).sum()) for n in names}
            text = "DIFFERENT TUNINGS: " + ",  ".join(f"{n} ({counts[n]} wafer{'s' if counts[n] != 1 else ''})" for n in names)
            self.fig.text(0.005, 0.005, text, ha="left", va="bottom", fontsize=8, color="#c62828",
                          fontweight="bold")
            self.tuning_warn.configure(text="The ticked wafers were analysed with different tunings: "
                                            "their numbers can't be compared directly.")
        else:
            self.fig.text(0.005, 0.005, f"tuning: {names[0] if names else wafer.tuning.UNKNOWN}", ha="left",
                          va="bottom", fontsize=8, color="#777")
            self.tuning_warn.configure(text="")
        versions = sorted(df["code_version"].dropna().astype(str).unique()) if "code_version" in df else []
        if len(versions) > 1:   # same parameters, but the analysis code changed in between
            self.fig.text(0.995, 0.005, "DIFFERENT CODE VERSIONS: " + ", ".join(versions), ha="right",
                          va="bottom", fontsize=8, color="#c62828", fontweight="bold")
            self.tuning_warn.configure(text=(self.tuning_warn.cget("text") + " " if len(names) > 1 else "")
                                       + "The ticked wafers were analysed with different code versions "
                                       f"({', '.join(versions)}; '?' = before versions were recorded): run "
                                       "them again to compare.")

    def _tol_lines(self, ax, param, axis="x"):
        m = param.metric if param is not None else None
        if m is None or not self.show_tol.get():
            return
        line = ax.axvline if axis == "x" else ax.axhline
        line(m.pass_max, color=COLORS["pass"], ls="--", lw=1.2, label=None)
        line(m.fail_min, color=COLORS["fail"], ls="--", lw=1.2, label=None)

    def _yield(self, ax, df, wafers, param):
        s = wafer.summary(df, param, self.tab.metrics, order=wafers)
        if self.sort_worst.get():
            s = s.sort_values(["fail_pct", "warn_pct"], ascending=False, kind="stable")
        labels = [self.short[n] for n in s["wafer"]]
        left = np.zeros(len(s))
        for st in STATUS_ORDER:
            pct = s[f"{st}_pct"].to_numpy()
            ax.barh(labels, pct, left=left, color=COLORS[st], edgecolor="white", label=NAMES[st])
            for i, (p, l0, n) in enumerate(zip(pct, left, s["devices"])):
                if p >= 6:
                    ax.text(l0 + p / 2, i, f"{round(p * n / 100)}", ha="center", va="center", fontsize=8,
                            color="white" if st in ("pass", "fail") else "black")
            left += pct
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of devices  (numbers in the bars: device counts)")
        ax.set_title(f"Yield per wafer: {param.label}")
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    def _curves(self, ax, df, wafers, param):
        vals = param.values(df)
        positive = vals[vals > 0]
        for n in wafers:
            v = np.sort(vals[df["wafer"] == n].dropna().to_numpy())
            if len(v) == 0:
                continue
            y = np.arange(1, len(v) + 1) / len(v) * 100
            ax.step(np.r_[v[0], v], np.r_[0, y], where="post", color=self.colors[n], lw=1.8,
                    label=f"{self.short[n]}  (median {fmt(float(np.median(v)))})")
        self._tol_lines(ax, param)
        if self.log_x.get() and len(positive):
            ax.set_xscale("symlog", linthresh=max(float(positive.min()), 1e-3))
        if vals.min() >= 0:
            ax.set_xlim(left=0)
        ax.set_ylim(0, 101)
        ax.set_xlabel(param.label)
        ax.set_ylabel("% of devices at or below")
        ax.set_title(f"Distribution per wafer: {param.label}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    def _wafer_map(self, ax, df, wafers, param):
        stat = self.map_stat.get()
        if stat == "% fail" and param.metric is None:
            stat = "mean"
            self.map_stat.set(stat)
        d = df.assign(_v=param.values(df)).dropna(subset=["x", "y"])
        if stat == "% fail":
            d = d.assign(_s=param.statuses(d, self.tab.metrics))
            d = d[d["_s"] != "nodata"]
            agg = d.groupby("device").agg(x=("x", "first"), y=("y", "first"),
                                          v=("_s", lambda s: 100.0 * (s == "fail").mean()), n=("_s", "size"))
            cmap, norm = cm.RdYlGn_r, mcolors.Normalize(0, 100)
        else:
            d = d.dropna(subset=["_v"])
            agg = d.groupby("device").agg(x=("x", "first"), y=("y", "first"), v=("_v", stat), n=("_v", "size"))
            vmax = float(agg["v"].max()) if len(agg) else 1.0
            vmin = float(agg["v"].min()) if len(agg) else 0.0
            cmap, norm = cm.viridis, mcolors.Normalize(vmin, vmax if vmax > vmin else vmin + 1)
        if agg.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data for this parameter.", ha="center", va="center", transform=ax.transAxes)
            return
        pos = np.array(list(self.tab.positions.values())) if self.tab.positions else agg[["x", "y"]].to_numpy()
        px, py = _pitch(pos[:, 0]), _pitch(pos[:, 1])
        flip = self.tab.flip_y.get()
        for dev, r in agg.iterrows():
            y = -r["y"] if not flip else r["y"]          # Y down unless flipped, as on the map tab
            x0, y0 = r["x"] - px * 0.45, y - py * 0.45
            color = cmap(norm(r["v"]))
            ax.add_patch(Rectangle((x0, y0), px * 0.9, py * 0.9, facecolor=color, edgecolor="#555"))
            dark = color[0] * 0.3 + color[1] * 0.6 + color[2] * 0.1 < 0.5
            ax.text(r["x"], y + py * 0.2, str(int(dev)), ha="center", va="center", fontsize=7,
                    color="white" if dark else "black")
            ax.text(r["x"], y - py * 0.12, f"{r['v']:.3g}", ha="center", va="center", fontsize=8,
                    weight="bold", color="white" if dark else "black")
            self._map_cells.append((x0, y0, x0 + px * 0.9, y0 + py * 0.9,
                                    f"device {int(dev)}: {stat} {fmt(float(r['v']))} over {int(r['n'])} wafer(s)"))
        ys = -pos[:, 1] if not flip else pos[:, 1]
        ax.set_xlim(pos[:, 0].min() - px, pos[:, 0].max() + px)
        ax.set_ylim(ys.min() - py, ys.max() + py)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        unit = "% of wafers failing" if stat == "% fail" else f"{stat} of {param.label}"
        self.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=unit, shrink=0.8)
        ax.set_title(f"Per position over {len(wafers)} wafer(s): {unit}")

    def _scatter(self, ax, df, wafers, px, py):
        if py is None:
            return
        xv, yv = px.values(df), py.values(df)
        for n in wafers:
            sel = (df["wafer"] == n) & xv.notna() & yv.notna()
            if not sel.any():
                continue
            r = xv[sel].corr(yv[sel], method="spearman") if sel.sum() > 2 else np.nan
            art = ax.scatter(xv[sel], yv[sel], s=28, color=self.colors[n], alpha=0.85, edgecolor="white",
                             linewidth=0.5, picker=5, label=f"{self.short[n]}  r = {r:.2f}  n = {int(sel.sum())}")
            self._picked[art] = df[sel]
        self._tol_lines(ax, px, "x")
        self._tol_lines(ax, py, "y")
        ax.set_xlabel(px.label)
        ax.set_ylabel(py.label)
        ax.set_title(f"{py.label} vs {px.label}  (r = Spearman rank correlation)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        self.status.configure(text="Click a point: which device. Double-click: open it.")

    def _fill_table(self, param):
        self.tree.delete(*self.tree.get_children())
        if self.summary is None:
            self.table_lbl.configure(text="Summary")
            return
        self.table_lbl.configure(text=f"Summary per wafer: {param.label}")
        for n in self.summary["wafer"]:
            self.tree.tag_configure(n, foreground=self.colors.get(n, "black"))
        for rec in self.summary.to_dict("records"):
            self.tree.insert("", "end", values=[rec[c] if c in TEXT_COLUMNS else fmt(rec.get(c, np.nan))
                                                for c, _, _ in SUMMARY_COLUMNS], tags=(rec["wafer"],))

    # ------------------------------------------------------------ events, export

    def _on_pick(self, event):
        rows = self._picked.get(event.artist)
        if rows is None or not len(event.ind):
            return
        row = rows.iloc[int(event.ind[0])].to_dict()
        x, y = self.params[self.x_var.get()], self.params[self.y_var.get()]
        self.status.configure(text=f"{row['wafer']}  device {int(row['device'])}:  {x.label} = "
                                   f"{fmt(x.values(rows).iloc[int(event.ind[0])])},  {y.label} = "
                                   f"{fmt(y.values(rows).iloc[int(event.ind[0])])}")
        if event.mouseevent.dblclick:
            self.tab.show_device(row)

    def _on_motion(self, event):
        if not self._map_cells or event.xdata is None:
            return
        for x0, y0, x1, y1, text in self._map_cells:
            if x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1:
                self.status.configure(text=text)
                return
        self.status.configure(text="")

    def _export(self):
        if self.summary is None:
            return
        p = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                         initialdir=self.tab.results_root, initialfile="wafer_summary.csv")
        if p:
            try:
                self.summary.round(4).to_csv(p, index=False)
            except PermissionError:
                messagebox.showerror("Export", f"Can't write {p}: is it open in Excel?", parent=self)
