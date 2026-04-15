"""
Author: Samuel Lehmann
Network with him at: https://www.linkedin.com/in/samuellehmann/

Modifications:
- Responsive layout (uses pack/grid frames instead of absolute .place coords)
- Three independent overlay lines: slope (with slope+intercept inputs),
  horizontal (y-value input), vertical (x-value input). Each is toggleable,
  draggable, and its input field updates live during drag.
- Scroll-wheel zoom + Reset Zoom button (with decade-snapping on log axes).
- Line equations adapt to axis mode (linear, semilog, log-log) so a "slope"
  always means "slope in the relevant log/linear coordinates".
"""

import math
import warnings
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.spatial import ConvexHull
import numpy as np
import Field
import Main

# Matplotlib whines when it encounters control characters (e.g. stray \r from
# CSV line endings) in text labels. The warning is cosmetic — silence it.
warnings.filterwarnings("ignore", message=r".*Glyph 13.*")


# ============================================================================
# Module-level state
# ============================================================================

# GUI core
window, figure, hover_label, mat_info_label = (None for _ in range(4))
canvas = None
x_combo, y_combo = None, None

# Slope-line widgets and state
slope_line_checked = None      # IntVar
slope_label = None             # Label
slope_field = None             # Entry (slope value)
intercept_label = None         # Label
intercept_field = None         # Entry (y-intercept in "effective" coords)
slope_line_obj = None          # Line2D currently on the axes, or None

# Horizontal-line widgets and state
hline_checked = None
hline_label = None
hline_field = None             # Entry (y value where the hline sits)
hline_obj = None

# Vertical-line widgets and state
vline_checked = None
vline_label = None
vline_field = None             # Entry (x value where the vline sits)
vline_obj = None

# Log-axis checkboxes
x_axis_checked, y_axis_checked = None, None

# Drag state
dragging_target = None         # 'slope', 'hline', 'vline', or None
_updating_field_programmatically = False  # suppress trace recursion during drag

# Callbacks and axis mode
update_graph_callback = None
LOG, SEMILOGX, SEMILOGY, LINEAR = range(0, 4)
current_axis_type = LINEAR
initialized = False

# Labels matplotlib uses to identify overlay lines (so hover/click skips them)
SLOPE_LINE_LABEL = "_overlay_slope_"
HLINE_LABEL      = "_overlay_hline_"
VLINE_LABEL      = "_overlay_vline_"
ALL_OVERLAY_LABELS = (SLOPE_LINE_LABEL, HLINE_LABEL, VLINE_LABEL)


# ============================================================================
# Small helpers
# ============================================================================

def _safe_float(field):
    """Return the float value of an Entry, or None if invalid/missing."""
    try:
        return float(field.get()) if field is not None else None
    except (ValueError, AttributeError):
        return None


def _format_number(v):
    """Format a number for display in a small input field."""
    if v == 0:
        return "0"
    return f"{v:.4g}"


def _set_field_programmatically(field, value):
    """Update an Entry's text without triggering its own trace handler.
    Used while dragging so the displayed value stays in sync without
    causing a recursive redraw."""
    global _updating_field_programmatically
    _updating_field_programmatically = True
    try:
        field.delete(0, tk.END)
        field.insert(0, _format_number(value))
    finally:
        _updating_field_programmatically = False


def _eff_x(v, is_log):
    """Convert a data x-value to its 'effective' coordinate
    (log10(v) on log axes, v itself on linear). Used by every line equation."""
    return math.log10(v) if is_log else v


def _eff_y(v, is_log):
    return math.log10(v) if is_log else v


# A single Entry subclass used by all three lines. Constructed with a
# callback that fires whenever the user types a valid float.
class RestrictedEntry(tk.Entry):
    def __init__(self, master, on_change=None, **kwargs):
        self.var = tk.StringVar()
        tk.Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.on_change = on_change
        self.var.trace("w", self._check)
        self.get, self.set = self.var.get, self.var.set

    def _check(self, *_args):
        # Skip if drag is updating us (the drag has already redrawn the line)
        if _updating_field_programmatically:
            return
        try:
            float(self.get())
        except ValueError:
            return  # leave the bad text alone; line stays at last valid value
        if self.on_change is not None:
            self.on_change()


# ============================================================================
# Window construction
# ============================================================================

def create_window(update_plot):
    global window, figure, canvas, x_combo, y_combo, update_graph_callback
    global slope_line_checked, slope_label, slope_field, intercept_label, intercept_field
    global hline_checked, hline_label, hline_field
    global vline_checked, vline_label, vline_field
    global x_axis_checked, y_axis_checked

    update_graph_callback = update_plot

    window = tk.Tk()
    window.title("OpenAshby: Material Property Charts")
    window.geometry("1100x950")
    window.configure(bg="White")

    # ---- Top control strip ----
    controls = tk.Frame(window, bg="White")
    controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
    # Centering trick: outer columns absorb slack, middle column holds content
    controls.columnconfigure(0, weight=1)
    controls.columnconfigure(1, weight=0)
    controls.columnconfigure(2, weight=1)

    filtered_fields = list(Field.FIELDS[Field.FIELDS["Graphable"]]["Text"])

    # Wrapper: replot, then snap axes to clean decades and redraw any active lines
    def _on_view_changed(*args):
        update_graph_callback(*args)
        reset_zoom()

    # Row 0: X-Axis dropdown
    x_row = tk.Frame(controls, bg="White")
    x_row.grid(row=0, column=1, pady=2)
    tk.Label(x_row, text="X-Axis:", bg="White").pack(side=tk.LEFT, padx=(0, 4))
    x_combo = ttk.Combobox(x_row, width=55, state="readonly", values=filtered_fields)
    x_combo.set(Main.DEFAULT_X_CAT)
    x_combo.pack(side=tk.LEFT)
    x_combo.bind("<<ComboboxSelected>>", _on_view_changed)

    # Row 1: Y-Axis dropdown
    y_row = tk.Frame(controls, bg="White")
    y_row.grid(row=1, column=1, pady=2)
    tk.Label(y_row, text="Y-Axis:", bg="White").pack(side=tk.LEFT, padx=(0, 4))
    y_combo = ttk.Combobox(y_row, width=55, state="readonly", values=filtered_fields)
    y_combo.set(Main.DEFAULT_Y_CAT)
    y_combo.pack(side=tk.LEFT)
    y_combo.bind("<<ComboboxSelected>>", _on_view_changed)

    # Row 2: Slope line — checkbox + slope + intercept inputs
    slope_row = tk.Frame(controls, bg="White")
    slope_row.grid(row=2, column=1, pady=(8, 2))
    slope_line_checked = tk.IntVar()
    tk.Checkbutton(slope_row, text="Add Slope Line (drag)",
                   variable=slope_line_checked, command=_slope_line_check_change,
                   bg="White").pack(side=tk.LEFT)
    slope_label = tk.Label(slope_row, text="Slope:", bg="White")
    slope_field = RestrictedEntry(slope_row, on_change=_redraw_slope_line, width=8)
    slope_field.insert(0, "1")
    intercept_label = tk.Label(slope_row, text="Intercept:", bg="White")
    intercept_field = RestrictedEntry(slope_row, on_change=_redraw_slope_line, width=10)
    # slope/intercept fields are shown only when the checkbox is on (see _slope_line_check_change)

    # Row 3: Horizontal line
    hline_row = tk.Frame(controls, bg="White")
    hline_row.grid(row=3, column=1, pady=2)
    hline_checked = tk.IntVar()
    tk.Checkbutton(hline_row, text="Add Horizontal Line (drag)",
                   variable=hline_checked, command=_hline_check_change,
                   bg="White").pack(side=tk.LEFT)
    hline_label = tk.Label(hline_row, text="Y:", bg="White")
    hline_field = RestrictedEntry(hline_row, on_change=_redraw_hline, width=12)

    # Row 4: Vertical line
    vline_row = tk.Frame(controls, bg="White")
    vline_row.grid(row=4, column=1, pady=2)
    vline_checked = tk.IntVar()
    tk.Checkbutton(vline_row, text="Add Vertical Line (drag)",
                   variable=vline_checked, command=_vline_check_change,
                   bg="White").pack(side=tk.LEFT)
    vline_label = tk.Label(vline_row, text="X:", bg="White")
    vline_field = RestrictedEntry(vline_row, on_change=_redraw_vline, width=12)

    # Row 5: Log axes + Reset Zoom
    log_row = tk.Frame(controls, bg="White")
    log_row.grid(row=5, column=1, pady=(6, 2))
    x_axis_checked = tk.IntVar()
    y_axis_checked = tk.IntVar()
    tk.Checkbutton(log_row, text="Log X", variable=x_axis_checked,
                   command=axis_check_change, bg="White").pack(side=tk.LEFT, padx=8)
    tk.Checkbutton(log_row, text="Log Y", variable=y_axis_checked,
                   command=axis_check_change, bg="White").pack(side=tk.LEFT, padx=8)
    tk.Button(log_row, text="Reset Zoom", command=reset_zoom,
              bg="White").pack(side=tk.LEFT, padx=12)

    # ---- Plot area ----
    plot_frame = tk.Frame(window, bg="White")
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    figure = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Mouse handlers
    figure.canvas.mpl_connect("button_press_event", _on_press)
    figure.canvas.mpl_connect("motion_notify_event", _on_motion)
    figure.canvas.mpl_connect("button_release_event", _on_release)
    figure.canvas.mpl_connect("scroll_event", _on_scroll)


# ============================================================================
# Checkbox handlers (show/hide each line's input fields and the line itself)
# ============================================================================

def _slope_line_check_change():
    if slope_line_checked.get():
        slope_label.pack(side=tk.LEFT, padx=(12, 4))
        slope_field.pack(side=tk.LEFT)
        intercept_label.pack(side=tk.LEFT, padx=(12, 4))
        intercept_field.pack(side=tk.LEFT)
        # On first turn-on, recenter the line and populate the intercept field
        _redraw_slope_line(recenter=True)
    else:
        _clear_slope_line()
        slope_field.pack_forget()
        slope_label.pack_forget()
        intercept_field.pack_forget()
        intercept_label.pack_forget()
        refresh()


def _hline_check_change():
    if hline_checked.get():
        hline_label.pack(side=tk.LEFT, padx=(12, 4))
        hline_field.pack(side=tk.LEFT)
        # Default y to the middle of the visible plot
        if _safe_float(hline_field) is None:
            primary = figure.gca()
            ylim = primary.get_ylim()
            is_log = primary.get_yscale() == "log"
            if is_log and ylim[0] > 0 and ylim[1] > 0:
                y_default = math.sqrt(ylim[0] * ylim[1])
            else:
                y_default = (ylim[0] + ylim[1]) / 2
            _set_field_programmatically(hline_field, y_default)
        _redraw_hline()
    else:
        _clear_hline()
        hline_field.pack_forget()
        hline_label.pack_forget()
        refresh()


def _vline_check_change():
    if vline_checked.get():
        vline_label.pack(side=tk.LEFT, padx=(12, 4))
        vline_field.pack(side=tk.LEFT)
        if _safe_float(vline_field) is None:
            primary = figure.gca()
            xlim = primary.get_xlim()
            is_log = primary.get_xscale() == "log"
            if is_log and xlim[0] > 0 and xlim[1] > 0:
                x_default = math.sqrt(xlim[0] * xlim[1])
            else:
                x_default = (xlim[0] + xlim[1]) / 2
            _set_field_programmatically(vline_field, x_default)
        _redraw_vline()
    else:
        _clear_vline()
        vline_field.pack_forget()
        vline_label.pack_forget()
        refresh()


def axis_check_change():
    """Toggle log/linear scaling on either axis, then replot + reset zoom."""
    global current_axis_type
    if x_axis_checked.get():
        current_axis_type = LOG if y_axis_checked.get() else SEMILOGX
    else:
        current_axis_type = SEMILOGY if y_axis_checked.get() else LINEAR
    if initialized:
        update_graph_callback()
        reset_zoom()


# ============================================================================
# Line plotting
# ============================================================================

def _clear_slope_line():
    global slope_line_obj
    if slope_line_obj is not None:
        try:
            slope_line_obj.remove()
        except Exception:
            pass
        slope_line_obj = None


def _clear_hline():
    global hline_obj
    if hline_obj is not None:
        try:
            hline_obj.remove()
        except Exception:
            pass
        hline_obj = None


def _clear_vline():
    global vline_obj
    if vline_obj is not None:
        try:
            vline_obj.remove()
        except Exception:
            pass
        vline_obj = None


def _redraw_slope_line(recenter=False):
    """Draw the slope line based on slope_field and intercept_field values.

    Equation in 'effective' coords:  y_eff = slope * x_eff + intercept
    where _eff is log10(value) on log axes, value otherwise. This gives:
      linear-linear  ->  y = m*x + b           (textbook line)
      semilog X      ->  y = m*log10(x) + b
      semilog Y      ->  y = 10**(m*x + b)
      log-log        ->  y = 10**b * x**m       (textbook Ashby selection line)
    """
    global slope_line_obj

    slope = _safe_float(slope_field)
    if slope is None:
        return

    primary = figure.gca()
    is_log_x = primary.get_xscale() == "log"
    is_log_y = primary.get_yscale() == "log"
    xlim = primary.get_xlim()
    ylim = primary.get_ylim()
    if is_log_x and (xlim[0] <= 0 or xlim[1] <= 0):
        return
    if is_log_y and (ylim[0] <= 0 or ylim[1] <= 0):
        return

    intercept = _safe_float(intercept_field)
    if recenter or intercept is None:
        x_mid = (_eff_x(xlim[0], is_log_x) + _eff_x(xlim[1], is_log_x)) / 2
        y_mid = (_eff_y(ylim[0], is_log_y) + _eff_y(ylim[1], is_log_y)) / 2
        intercept = y_mid - slope * x_mid
        _set_field_programmatically(intercept_field, intercept)

    _clear_slope_line()

    n = 300
    if is_log_x:
        x_data = np.logspace(math.log10(xlim[0]), math.log10(xlim[1]), n)
        x_eff = np.log10(x_data)
    else:
        x_data = np.linspace(xlim[0], xlim[1], n)
        x_eff = x_data
    y_eff = slope * x_eff + intercept
    if is_log_y:
        y_data = np.power(10.0, np.clip(y_eff, -300, 300))
    else:
        y_data = y_eff

    line, = primary.plot(x_data, y_data, linestyle="dotted", color="gray",
                         linewidth=2, picker=5, label=SLOPE_LINE_LABEL)
    slope_line_obj = line
    primary.set_xlim(xlim)
    primary.set_ylim(ylim)
    refresh()


def _redraw_hline():
    """Draw a horizontal line at the y-value in hline_field."""
    global hline_obj
    y = _safe_float(hline_field)
    if y is None:
        return
    primary = figure.gca()
    if primary.get_yscale() == "log" and y <= 0:
        return
    _clear_hline()
    hline_obj = primary.axhline(y, linestyle="--", color="darkblue",
                                 linewidth=2, picker=5, label=HLINE_LABEL)
    refresh()


def _redraw_vline():
    """Draw a vertical line at the x-value in vline_field."""
    global vline_obj
    x = _safe_float(vline_field)
    if x is None:
        return
    primary = figure.gca()
    if primary.get_xscale() == "log" and x <= 0:
        return
    _clear_vline()
    vline_obj = primary.axvline(x, linestyle="--", color="darkred",
                                 linewidth=2, picker=5, label=VLINE_LABEL)
    refresh()


# ============================================================================
# Mouse handlers (drag any of the three lines, scroll to zoom)
# ============================================================================

def _on_press(event):
    """Begin dragging whichever overlay line was clicked. First hit wins
    if multiple lines overlap at the click point."""
    global dragging_target
    if event.inaxes is None:
        return
    if (slope_line_obj is not None and slope_line_checked is not None
            and slope_line_checked.get()):
        if slope_line_obj.contains(event)[0]:
            dragging_target = "slope"
            return
    if (hline_obj is not None and hline_checked is not None
            and hline_checked.get()):
        if hline_obj.contains(event)[0]:
            dragging_target = "hline"
            return
    if (vline_obj is not None and vline_checked is not None
            and vline_checked.get()):
        if vline_obj.contains(event)[0]:
            dragging_target = "vline"
            return


def _on_motion(event):
    """While dragging, update the active line's value and its input field."""
    if dragging_target is None:
        return
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return
    primary = figure.gca()
    is_log_x = primary.get_xscale() == "log"
    is_log_y = primary.get_yscale() == "log"

    if dragging_target == "slope":
        if is_log_x and event.xdata <= 0:
            return
        if is_log_y and event.ydata <= 0:
            return
        slope = _safe_float(slope_field)
        if slope is None:
            return
        x_eff = _eff_x(event.xdata, is_log_x)
        y_eff = _eff_y(event.ydata, is_log_y)
        new_intercept = y_eff - slope * x_eff
        _set_field_programmatically(intercept_field, new_intercept)
        _redraw_slope_line()  # don't recenter — keep the dragged intercept
    elif dragging_target == "hline":
        if is_log_y and event.ydata <= 0:
            return
        _set_field_programmatically(hline_field, event.ydata)
        _redraw_hline()
    elif dragging_target == "vline":
        if is_log_x and event.xdata <= 0:
            return
        _set_field_programmatically(vline_field, event.xdata)
        _redraw_vline()


def _on_release(event):
    global dragging_target
    dragging_target = None


# ----- Scroll-wheel zoom -----------------------------------------------------

def _zoom_axis_limits(lim, scale, cursor, factor):
    """Return new (lo, hi) for an axis after zooming by `factor` toward `cursor`.
    Works in log space when scale == 'log' so visual zoom feels uniform on
    log-log plots (one scroll tick = same fraction of a decade)."""
    if scale == "log":
        if lim[0] <= 0 or lim[1] <= 0 or cursor <= 0:
            return lim
        ll, lh, lc = math.log10(lim[0]), math.log10(lim[1]), math.log10(cursor)
        return (10 ** (lc + (ll - lc) * factor),
                10 ** (lc + (lh - lc) * factor))
    new_lo = cursor + (lim[0] - cursor) * factor
    new_hi = cursor + (lim[1] - cursor) * factor
    return (new_lo, new_hi)


def _on_scroll(event):
    """Mouse-wheel zoom centered on the cursor. Up = in, Down = out."""
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return
    primary = figure.gca()
    factor = 1 / 1.2 if event.button == "up" else 1.2
    primary.set_xlim(_zoom_axis_limits(primary.get_xlim(),
                                       primary.get_xscale(),
                                       event.xdata, factor))
    primary.set_ylim(_zoom_axis_limits(primary.get_ylim(),
                                       primary.get_yscale(),
                                       event.ydata, factor))
    _redraw_active_lines()
    refresh()


# ============================================================================
# Reset Zoom
# ============================================================================

def _snap_to_decades(lim):
    """Round (lo, hi) outward to enclosing powers of 10. The closest log-axis
    analog of 'anchor at the origin'."""
    lo = max(lim[0], 1e-12)
    hi = max(lim[1], lo * 10)
    return (10 ** math.floor(math.log10(lo)),
            10 ** math.ceil(math.log10(hi)))


def _redraw_active_lines():
    """Re-render any overlay lines whose checkboxes are currently on."""
    if slope_line_checked is not None and slope_line_checked.get():
        _redraw_slope_line()
    if hline_checked is not None and hline_checked.get():
        _redraw_hline()
    if vline_checked is not None and vline_checked.get():
        _redraw_vline()


def reset_zoom():
    """Reset axis bounds to fit all plotted data, snapping to decades on log
    axes for a clean look. Preserves all overlay lines."""
    primary = figure.gca()
    primary.relim()
    primary.autoscale()
    if primary.get_xscale() == "log":
        primary.set_xlim(_snap_to_decades(primary.get_xlim()))
    if primary.get_yscale() == "log":
        primary.set_ylim(_snap_to_decades(primary.get_ylim()))
    primary.set_aspect("auto")
    _redraw_active_lines()
    refresh()


# ============================================================================
# Plot building (called from Main.py)
# ============================================================================

def identify_hovered_node(event):
    """Find a data point under the cursor. Skips overlay lines so they
    don't get mistaken for materials."""
    for line in figure.gca().lines:
        if line.get_label() in ALL_OVERLAY_LABELS:
            continue
        cont, ind = line.contains(event)
        if cont:
            x, y = line.get_data()
            x = x[ind["ind"][0]]
            y = y[ind["ind"][0]]
            return x, y
    return None


def build_figure(x_cat, y_cat, hover_event=None, click_event=None):
    """Clear and prepare the figure for new data."""
    global hover_label, mat_info_label, initialized
    global slope_line_obj, hline_obj, vline_obj
    # The previous line objects belong to the old (about-to-be-cleared) axes
    slope_line_obj = None
    hline_obj = None
    vline_obj = None

    figure.clf()

    if hover_event:
        figure.canvas.mpl_connect("motion_notify_event", hover_event)
        figure.canvas.mpl_connect("button_press_event", click_event)

    plt.xlabel(x_cat)
    plt.ylabel(y_cat)
    plt.title(y_cat + " As a Function Of " + x_cat)

    hover_label = figure.get_axes()[0].annotate(
        "", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"), ha="center")
    hover_label.set_visible(False)

    mat_info_label = figure.get_axes()[0].text(
        1, 1, "", transform=figure.get_axes()[0].transAxes,
        horizontalalignment="right", verticalalignment="top",
        bbox=dict(boxstyle="round", fc="w"))
    mat_info_label.set_visible(False)

    plt.grid()
    initialized = True
    # Overlay lines are drawn later by reset_zoom() (called by the dropdown
    # / log toggle wrapper after data has been plotted).


def add_to_plot(data_points, category):
    """Add datapoints to the plot, generates convex hulls, labels them."""
    if current_axis_type == LOG:
        plt.loglog(data_points[:, 0], data_points[:, 1], label=category,
                   linestyle="", marker="o")
    elif current_axis_type == SEMILOGX:
        plt.semilogx(data_points[:, 0], data_points[:, 1], label=category,
                     linestyle="", marker="o")
        plt.ylim(bottom=0)
    elif current_axis_type == SEMILOGY:
        plt.semilogy(data_points[:, 0], data_points[:, 1], label=category,
                     linestyle="", marker="o")
        plt.xlim(left=0)
    elif current_axis_type == LINEAR:
        plt.plot(data_points[:, 0], data_points[:, 1], label=category,
                 linestyle="", marker="o")
        plt.xlim(left=0)
        plt.ylim(bottom=0)

    colour = plt.gca().lines[-1].get_color()

    if len(data_points) >= 3:
        try:
            hull = ConvexHull(data_points)
            hp = [data_points[hull.vertices, 0], data_points[hull.vertices, 1]]
            hp[0] = np.concatenate((hp[0], [hp[0][0]]))
            hp[1] = np.concatenate((hp[1], [hp[1][0]]))
            plt.plot(hp[0], hp[1], color=colour)
            plt.fill_between(hp[0], hp[1], alpha=0.1, color=colour)
        except Exception:
            pass

    label_text = category
    if len(label_text) > 8:
        split_index = label_text.find(" ", 5, -1)
        if split_index > 0:
            label_text = label_text[:split_index] + "\n" + label_text[split_index:]
    plt.annotate(label_text,
                 [np.median(data_points[:, 0]), np.median(data_points[:, 1])],
                 color=colour, weight="bold", backgroundcolor="#ffffff80",
                 fontsize=8, va="center", ha="center")

    figure.gca().relim()
    figure.gca().autoscale()


def refresh():
    figure.canvas.draw()
    figure.canvas.flush_events()


def create_annotation(text, x, y):
    hover_label.xy = (x, y)
    hover_label.set_text(text)
    hover_label.get_bbox_patch().set_alpha(1)
    hover_label.set_visible(True)
    figure.canvas.draw_idle()


def create_mat_details(text):
    mat_info_label.set_text(text)
    mat_info_label.get_bbox_patch().set_alpha(1)
    mat_info_label.set_visible(True)


def get_selected_categories():
    return x_combo.get(), y_combo.get()


def hide_hover_annotation():
    hover_label.set_visible(False)


def hide_mat_details():
    mat_info_label.set_visible(False)


def start_loop():
    # Snap initial plot to clean axes; Main.generate_plot has already populated
    # the figure synchronously by this point.
    try:
        reset_zoom()
    except Exception:
        pass
    window.mainloop()
