"""Inventory tab: the wafers we have and their status, planning which two items to stack (so their
defective devices overlap), and the stacks made in production. Data in inventory.py."""
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np

from . import inventory, wafer
from .gui import loading, status
from .gui_wafer import COLORS, LIGHT, NAMES, TEXT_ON, _pitch

ROW_COLORS = {inventory.IN_STOCK: "#ffffff", inventory.RELEASED: "#dcf2dc", inventory.REJECTED: "#fadbd8",
              inventory.STACKED: "#e3e8f4", inventory.AVAILABLE: "#dcf2dc"}
# combined map of two items: which one loses the stacked device
PAIR_COLORS = {"good": "#43a047", "a": "#e53935", "b": "#fb8c00", "both": "#616161", "nodata": "#bdbdbd"}
PAIR_NAMES = {"good": "good in both", "a": "defect only in A", "b": "defect only in B",
              "both": "defect in both (overlap)", "nodata": "no data"}
PAIR_TEXT = {"good": "white", "a": "white", "b": "black", "both": "white", "nodata": "black"}
ALL = "All"


class DeviceMap(tk.Canvas):
    """Small wafer map: one coloured box per device at its position, with its number."""

    def __init__(self, master, positions, on_hover=None):
        super().__init__(master, background="white", highlightthickness=0, height=260)
        self.positions = positions
        self.on_hover = on_hover
        self.data = {}   # device -> (fill, text colour, hover text)
        self.bind("<Configure>", lambda e: self.draw())

    def show(self, data):
        self.data = data
        self.draw()

    def draw(self):
        self.delete("all")
        pos = self.positions or {d: (float(d), 0.0) for d in self.data}
        if not pos:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        xs, ys = np.array([p[0] for p in pos.values()]), np.array([p[1] for p in pos.values()])
        px, py = _pitch(xs), _pitch(ys)
        span_x, span_y = xs.max() - xs.min() + px, ys.max() - ys.min() + py
        s = min((cw - 16) / span_x, (ch - 16) / span_y)
        ox, oy = (cw - span_x * s) / 2, (ch - span_y * s) / 2
        w, h = px * s * 0.45, py * s * 0.45
        font = ("Segoe UI", -max(7, int(min(w, h) * 0.8)), "bold")
        for dev, (x, y) in pos.items():
            fill, fg, tip = self.data.get(dev, (COLORS["nodata"], "black", f"Device {dev}: no data"))
            gx, gy = ox + (x - xs.min() + px / 2) * s, oy + (y - ys.min() + py / 2) * s
            tag = f"dev{dev}"
            self.create_rectangle(gx - w, gy - h, gx + w, gy + h, fill=fill, outline="#555", tags=(tag,))
            self.create_text(gx, gy, text=str(dev), fill=fg, font=font, tags=(tag,))
            if self.on_hover:
                self.tag_bind(tag, "<Enter>", lambda e, t=tip: self.on_hover(t))
                self.tag_bind(tag, "<Leave>", lambda e: self.on_hover(""))


class InventoryTab(ttk.Frame):
    """`plates` from measurements.load(); show_wafer(plate, wafer) brings up that wafer's map."""

    def __init__(self, master, plates, show_wafer=None):
        super().__init__(master)
        self.plates, self.show_wafer = plates, show_wafer
        self.built = False
        self.wafers = self.stacks = None
        try:
            self.positions = wafer.load_positions(wafer.DEFAULT_POSITIONS)
        except Exception:
            self.positions = {}
        self.bind("<Map>", lambda e: self._shown())

    def _shown(self):
        if not self.built:
            self.built = True
            with loading("Loading the inventory..."):
                self._build()
            self.refresh()
        else:
            self.reload()   # pick up what others changed meanwhile

    # ------------------------------------------------------------ data

    @property
    def devices(self):
        return sorted(self.positions) if self.positions else None

    def _devices_of(self, *statuses):
        """The device map's devices, or (no map) those with results in any of `statuses`."""
        return self.devices or sorted(set().union(*statuses))

    def refresh(self):
        """Look for new wafers, re-read results and tolerances, show everything again."""
        if not self.built:
            return
        try:
            with loading("Looking for new wafers..."):
                new = inventory.sync(self.plates, progress=status)
        except Exception as e:
            messagebox.showerror("Inventory", f"Could not update the inventory:\n{e}")
            new = []
        inventory.clear_cache()
        self.reload()
        if new:
            status(f"New in the inventory: {', '.join(new)}")

    def reload(self):
        try:
            self.wafers, self.stacks = inventory.load()
        except Exception as e:
            messagebox.showerror("Inventory", f"Could not read {inventory.FOLDER}:\n{e}")
            return
        with loading("Reading the wafers' results..."):
            self._fill_wafers()
            self._fill_stacks()
            self._fill_items()
            self._show_pair()

    def _status(self, item):
        return inventory.item_status(item, self.wafers, self.stacks, self.plates)

    def _defective(self):
        return {"fail", "warn"} if self.warn_bad.get() else inventory.DEFECTIVE

    def _write(self, fn, *args):
        """Run an inventory change; show the error if it fails. True if it worked."""
        try:
            fn(*args)
        except (OSError, ValueError) as e:
            messagebox.showerror("Inventory", str(e))
            return False
        self.reload()
        return True

    # ------------------------------------------------------------ layout

    def _build(self):
        self.notebook = ttk.Notebook(self, style="Type.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        self.warn_bad = tk.BooleanVar(value=False)
        for title, build in (("Wafers", self._build_wafers), ("Plan a stack", self._build_plan),
                             ("Stacks", self._build_stacks)):
            frame = ttk.Frame(self.notebook, padding=6)
            self.notebook.add(frame, text=f"  {title}  ")
            build(frame)

    def _tree(self, master, columns, **kw):
        frame = ttk.Frame(master)
        tree = ttk.Treeview(frame, columns=[c for c, _, _ in columns], **kw)
        for c, text, width in columns:
            tree.heading(c, text=text)
            tree.column(c, width=width, anchor="w" if width > 90 else "center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        for s, color in ROW_COLORS.items():
            tree.tag_configure(s, background=color)
        return frame, tree

    def _build_wafers(self, master):
        bar = ttk.Frame(master)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="Refresh (find new wafers)", command=self.refresh).pack(side="left")
        ttk.Label(bar, text="Plate:").pack(side="left", padx=(16, 2))
        self.plate_filter = tk.StringVar(value=ALL)
        box = ttk.Combobox(bar, textvariable=self.plate_filter, state="readonly", width=8,
                           values=[ALL, *self.plates])
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._fill_wafers())
        ttk.Label(bar, text="Status:").pack(side="left", padx=(12, 2))
        self.status_filter = tk.StringVar(value=ALL)
        box = ttk.Combobox(bar, textvariable=self.status_filter, state="readonly", width=10,
                           values=[ALL, *inventory.WAFER_STATUSES])
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self._fill_wafers())
        ttk.Button(bar, text="Notes...", command=lambda: self._notes("wafer", self.wafer_tree)).pack(side="right")
        for text, st in (("Back to stock", inventory.IN_STOCK), ("Reject", inventory.REJECTED),
                         ("Release for stacking", inventory.RELEASED)):
            ttk.Button(bar, text=text, command=lambda st=st: self._set_status(st)).pack(side="right", padx=(0, 6))

        cols = [("plate", "Plate", 60), ("status", "Status", 80), ("pass", "Pass", 50), ("warn", "Warn", 50),
                ("fail", "Fail", 50), ("nodata", "No data", 60), ("good", "Good %", 60),
                ("stack", "Stack", 70), ("changed", "Status changed", 150), ("notes", "Notes", 300)]
        frame, self.wafer_tree = self._tree(master, cols, show="tree headings", selectmode="extended")
        self.wafer_tree.heading("#0", text="Wafer")
        self.wafer_tree.column("#0", width=260)
        frame.pack(fill="both", expand=True)
        self.wafer_tree.bind("<Double-Button-1>", lambda e: self._open_wafer())
        ttk.Label(master, foreground="#666", font=("Segoe UI", 8), text=(
            "Device counts: worst status over the plate's measurement types with results, using their saved "
            "tolerances. Good % = devices not failing, of the devices on the map. Double-click a wafer to "
            "open its wafer map. Released wafers can be chosen in 'Plan a stack'.")).pack(anchor="w", pady=(4, 0))

    def _build_plan(self, master):
        bar = ttk.Frame(master)
        bar.pack(fill="x")
        self.item_a, self.item_b = tk.StringVar(), tk.StringVar()
        self.item_boxes = []
        for text, var in (("A:", self.item_a), ("B:", self.item_b)):
            ttk.Label(bar, text=text, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 2))
            box = ttk.Combobox(bar, textvariable=var, state="readonly", width=30)
            box.pack(side="left", padx=(0, 12))
            box.bind("<<ComboboxSelected>>", lambda e: self._show_pair())
            self.item_boxes.append(box)
        self.with_stock = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Also wafers in stock", variable=self.with_stock,
                        command=self._fill_items).pack(side="left")
        ttk.Checkbutton(bar, text="Count warn as defective", variable=self.warn_bad,
                        command=self._show_pair).pack(side="left", padx=(12, 0))

        bar2 = ttk.Frame(master)
        bar2.pack(fill="x", pady=(6, 0))
        ttk.Button(bar2, text="Best partners for A", command=self._best_partner).pack(side="left")
        ttk.Button(bar2, text="Suggest pairing of all", command=self._suggest_all).pack(side="left", padx=6)
        ttk.Button(bar2, text="Record A + B as stacked...",
                   command=lambda: self._add_stack(self.item_a.get(), self.item_b.get())).pack(side="right")

        maps = ttk.Frame(master)
        maps.pack(fill="both", expand=True, pady=(8, 0))
        self.map_titles, self.maps = [], []
        self.hover = ttk.Label(master, foreground="#444", font=("Consolas", 9))
        for col in range(3):
            t = ttk.Label(maps, font=("Segoe UI", 10, "bold"))
            t.grid(row=0, column=col, sticky="w", padx=4)
            m = DeviceMap(maps, self.positions, on_hover=lambda t: self.hover.configure(text=t))
            m.grid(row=1, column=col, sticky="nsew", padx=4)
            maps.columnconfigure(col, weight=1)
            self.map_titles.append(t)
            self.maps.append(m)
        maps.rowconfigure(1, weight=1)
        legend = ttk.Frame(maps)
        legend.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        for k in ("pass", "warn", "fail", "nodata"):
            tk.Label(legend, text=f" {NAMES[k]} ", background=COLORS[k], fg=TEXT_ON[k],
                     font=("Segoe UI", 8, "bold")).pack(side="left", padx=1)
        ttk.Label(legend, text="   stacked:").pack(side="left")
        for k in PAIR_COLORS:
            tk.Label(legend, text=f" {PAIR_NAMES[k]} ", background=PAIR_COLORS[k], fg=PAIR_TEXT[k],
                     font=("Segoe UI", 8, "bold")).pack(side="left", padx=1)
        self.hover.pack(anchor="w", pady=(2, 0))
        self.pair_lbl = ttk.Label(master, font=("Segoe UI", 10, "bold"))
        self.pair_lbl.pack(anchor="w", pady=(4, 4))

        cols = [("b", "B", 260), ("good", "Good stacked", 90), ("yield", "Yield %", 70),
                ("a_only", "Lost by A", 80), ("b_only", "Lost by B", 80), ("both", "Overlapping defects", 120)]
        frame, self.pair_tree = self._tree(master, cols, show="tree headings", selectmode="browse", height=8)
        self.pair_tree.heading("#0", text="A")
        self.pair_tree.column("#0", width=260)
        frame.pack(fill="both")
        self.pair_tree.bind("<<TreeviewSelect>>", lambda e: self._pair_chosen())
        ttk.Label(master, foreground="#666", font=("Segoe UI", 8), text=(
            "A stacked device is good only if it is good in both items, so the best pairs are those whose "
            "defective devices lie on the same positions. Click a suggestion to show it above.")
                  ).pack(anchor="w", pady=(4, 0))

    def _build_stacks(self, master):
        bar = ttk.Frame(master)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="Add stack (stacked in production)...",
                   command=lambda: self._add_stack("", "")).pack(side="left")
        ttk.Button(bar, text="Notes...", command=lambda: self._notes("stack", self.stack_tree)).pack(side="right")
        cols = [("status", "Status", 80), ("date", "Stacked on", 130), ("by", "By", 80),
                ("good", "Good devices", 90), ("bad", "Defective", 80), ("parent", "Used in", 70),
                ("notes", "Notes", 300)]
        frame, self.stack_tree = self._tree(master, cols, show="tree headings", selectmode="browse")
        self.stack_tree.heading("#0", text="Stack / items")
        self.stack_tree.column("#0", width=300)
        frame.pack(fill="both", expand=True)
        ttk.Label(master, foreground="#666", font=("Segoe UI", 8), text=(
            "Good / defective devices are those predicted when the stack was recorded. An available stack can "
            "be stacked again with another wafer or stack.")).pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------ filling

    def _fill_wafers(self):
        t = self.wafer_tree
        t.delete(*t.get_children())
        if self.wafers is None:
            return
        for r in self.wafers.to_dict("records"):
            if self.plate_filter.get() not in (ALL, r["plate"]) or self.status_filter.get() not in (ALL, r["status"]):
                continue
            st = self._status(r["wafer"])
            devs = self._devices_of(st)
            c = inventory.counts(st, devs)
            good = f"{100 * (c['pass'] + c['warn']) / len(devs):.0f}" if devs and st else "-"
            changed = f"{r['status_changed']} {r['changed_by']}".strip()
            t.insert("", "end", iid=r["wafer"], text=r["wafer"], tags=(r["status"],), values=(
                r["plate"], r["status"], c["pass"], c["warn"], c["fail"], c["nodata"], good, r["stack"],
                changed, r["notes"]))

    def _fill_stacks(self):
        t = self.stack_tree
        t.delete(*t.get_children())
        if self.stacks is None:
            return
        by_name = {r["stack"]: r for r in self.stacks.to_dict("records")}

        def add(parent, item, iid):
            r = by_name.get(item)
            if r is None:   # a wafer
                w = self.wafers[self.wafers["wafer"] == item]
                plate = w.iloc[0]["plate"] if len(w) else "?"
                t.insert(parent, "end", iid=iid, text=f"{item}  ({plate})", values=("wafer",))
                return
            n = lambda s: len([d for d in s.split(";") if d])
            t.insert(parent, "end", iid=iid, text=item, open=parent == "", tags=(r["status"],), values=(
                r["status"], r["stacked_on"], r["created_by"], n(r["good_devices"]), n(r["defective_devices"]),
                r["parent"], r["notes"]))
            add(iid, r["item_a"], f"{iid}/a")
            add(iid, r["item_b"], f"{iid}/b")

        for r in by_name.values():
            if not r["parent"]:   # substacks are shown inside their stack
                add("", r["stack"], r["stack"])

    def _fill_items(self):
        if self.wafers is None:
            return
        items = self._candidates()
        for box in self.item_boxes:
            box.configure(values=items)
        for var in (self.item_a, self.item_b):
            if var.get() not in items:
                var.set("")

    def _item_text(self, item):
        if item in set(self.stacks["stack"]):
            return f"Stack {item} ({' + '.join(inventory.wafers_of(item, self.stacks))})"
        w = self.wafers[self.wafers["wafer"] == item]
        return f"{item} ({w.iloc[0]['status']})" if len(w) else item

    def _show_pair(self):
        if self.wafers is None:
            return
        a, b = self.item_a.get(), self.item_b.get()
        sa, sb = (self._status(a) if a else {}), (self._status(b) if b else {})
        devs = self._devices_of(sa, sb)
        bad = self._defective()
        for i, (item, st) in enumerate(((a, sa), (b, sb))):
            c = inventory.counts(st, devs)
            self.map_titles[i].configure(text=f"{'AB'[i]}: {self._item_text(item)}   "
                                              f"pass {c['pass']}  warn {c['warn']}  fail {c['fail']}" if item
                                         else f"{'AB'[i]}: choose an item")
            self.maps[i].show({d: (COLORS[s], TEXT_ON[s], f"{'AB'[i]} device {d}: {NAMES[s]}")
                               for d, s in st.items()})
        if not (a and b) or a == b:
            self.map_titles[2].configure(text="A + B stacked")
            self.maps[2].show({})
            self.pair_lbl.configure(text="Choose two different items A and B.")
            return
        cats, n = inventory.compare(sa, sb, devs, bad)
        self.map_titles[2].configure(text="A + B stacked")
        self.maps[2].show({d: (PAIR_COLORS[c], PAIR_TEXT[c], f"Stacked device {d}: {PAIR_NAMES[c]}   "
                               f"(A {NAMES[sa.get(d, 'nodata')]}, B {NAMES[sb.get(d, 'nodata')]})")
                           for d, c in cats.items()})
        self.pair_lbl.configure(text=(
            f"Stacked: {n['good']} good of {len(devs)} devices ({100 * n['good'] / max(len(devs), 1):.0f} %).   "
            f"Lost by A only {n['a']}, by B only {n['b']}, overlapping defects {n['both']}, no data {n['nodata']}"))

    def _fill_pairs(self, ranked):
        t = self.pair_tree
        t.delete(*t.get_children())
        devs = self.devices
        for i, (a, b, n) in enumerate(ranked):
            total = len(devs) if devs else sum(n.values())
            t.insert("", "end", iid=str(i), text=a, values=(b, n["good"], f"{100 * n['good'] / max(total, 1):.0f}",
                                                             n["a"], n["b"], n["both"]))

    def _candidates(self):
        items = inventory.available(self.wafers, self.stacks)
        if self.with_stock.get():
            items += self.wafers.loc[self.wafers["status"] == inventory.IN_STOCK, "wafer"].tolist()
        return items

    def _ranked(self, partner_of=None):
        items = self._candidates()
        devs = self.devices or sorted(set().union(*(self._status(i) for i in items + [partner_of] if i)))
        with loading("Comparing the items..."):
            return inventory.rank_pairs(items, self.wafers, self.stacks, devs, self._defective(), self.plates,
                                        partner_of=partner_of)

    def _best_partner(self):
        if not self.item_a.get():
            messagebox.showinfo("Best partners", "Choose item A first.")
            return
        self._fill_pairs(self._ranked(self.item_a.get()))

    def _suggest_all(self):
        pairs = inventory.best_pairing(self._ranked())
        if not pairs:
            messagebox.showinfo("Suggest pairing", "Fewer than two released wafers or available stacks.")
        self._fill_pairs(pairs)

    def _pair_chosen(self):
        sel = self.pair_tree.selection()
        if sel:
            item = self.pair_tree.item(sel[0])
            self.item_a.set(item["text"])
            self.item_b.set(item["values"][0])
            self._show_pair()

    # ------------------------------------------------------------ actions

    def _selected_wafers(self):
        return list(self.wafer_tree.selection())

    def _set_status(self, st):
        names = self._selected_wafers()
        if not names:
            messagebox.showinfo("Inventory", "Select one or more wafers first.")
            return
        stacked = [n for n in names if self.wafers.loc[self.wafers["wafer"] == n, "status"].iloc[0]
                   == inventory.STACKED]
        if stacked:
            messagebox.showerror("Inventory", f"Already in a stack: {', '.join(stacked)}")
            return
        self._write(inventory.set_status, names, st)

    def _notes(self, kind, tree):
        sel = tree.selection()
        df, col = (self.wafers, "wafer") if kind == "wafer" else (self.stacks, "stack")
        if not sel or sel[0] not in set(df[col]):
            messagebox.showinfo("Notes", f"Select a {kind} first.")
            return
        old = df.loc[df[col] == sel[0], "notes"].iloc[0]
        new = simpledialog.askstring("Notes", f"Notes for {sel[0]}:", initialvalue=old, parent=self)
        if new is not None:
            self._write(inventory.set_notes, kind, sel[0], new)

    def _open_wafer(self):
        sel = self._selected_wafers()
        if sel and self.show_wafer:
            plate = self.wafers.loc[self.wafers["wafer"] == sel[0], "plate"].iloc[0]
            if not self.show_wafer(plate, sel[0]):
                messagebox.showinfo("Inventory", f"{sel[0]} has no results yet: run the batch on it first.")

    def _add_stack(self, a, b):
        if self.wafers is None:
            return
        items = inventory.available(self.wafers, self.stacks) + \
            self.wafers.loc[self.wafers["status"] == inventory.IN_STOCK, "wafer"].tolist()
        AddStackDialog(self, items, a, b)

    def record_stack(self, name, a, b, date, notes):
        """From AddStackDialog. True if it was recorded."""
        not_released = [i for i in (a, b) if i in set(self.wafers["wafer"])
                        and self.wafers.loc[self.wafers["wafer"] == i, "status"].iloc[0] != inventory.RELEASED]
        if not_released and not messagebox.askyesno(
                "Add stack", f"Not released for stacking: {', '.join(not_released)}.\nRecord the stack anyway?",
                parent=self):
            return False
        devs = self._devices_of(self._status(a), self._status(b))
        if not self._write(lambda: inventory.add_stack(name, a, b, devs, date, notes, plates=self.plates,
                                                       defective=self._defective())):
            return False
        self.notebook.select(2)
        return True


class AddStackDialog(tk.Toplevel):
    def __init__(self, tab, items, a="", b=""):
        super().__init__(tab)
        self.tab = tab
        self.title("Add stack (stacked in production)")
        self.transient(tab.winfo_toplevel())
        f = ttk.Frame(self, padding=12)
        f.pack(fill="both", expand=True)
        self.vars = {}
        fields = [("Name", inventory.next_stack_name(tab.stacks), None), ("Item A", a, items),
                  ("Item B", b, items), ("Stacked on", inventory.now(), None), ("Notes", "", None)]
        for row, (label, value, values) in enumerate(fields):
            ttk.Label(f, text=label + ":").grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=value)
            w = (ttk.Combobox(f, textvariable=var, values=values, state="readonly", width=40) if values is not None
                 else ttk.Entry(f, textvariable=var, width=43))
            w.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
            self.vars[label] = var
        ttk.Label(f, foreground="#666", font=("Segoe UI", 8), wraplength=380, justify="left", text=(
            "Items are released wafers, available stacks or wafers in stock. Their status becomes 'stacked'."
        )).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(6, 0))
        btns = ttk.Frame(f)
        btns.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Add stack", command=self._ok).pack(side="right", padx=6)
        self.grab_set()

    def _ok(self):
        v = {k: var.get().strip() for k, var in self.vars.items()}
        if not v["Item A"] or not v["Item B"]:
            messagebox.showerror("Add stack", "Choose both items.", parent=self)
            return
        if self.tab.record_stack(v["Name"], v["Item A"], v["Item B"], v["Stacked on"], v["Notes"]):
            self.destroy()
