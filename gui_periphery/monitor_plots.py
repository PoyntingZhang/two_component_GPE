"""CPU-only monitor figures. Raw saved diagnostics and solver units are unchanged."""

from io import StringIO
from pathlib import Path
import threading

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
import numpy as np


# Matplotlib's font machinery is shared by HTTP threads.
PLOT_LOCK = threading.Lock()
BEC = LinearSegmentedColormap.from_list(
    "BEC", np.loadtxt(Path(__file__).with_name("bec_colormap.txt"), skiprows=1))
PHASE = LinearSegmentedColormap.from_list("original_phase", np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]) / 255)

# hbar^2 / (2 m ell^2 k_B), with ell = 1 micrometre and m = 38.9637064864 u (39K).
ENERGY_NK_39K = 6.22486135015065


def scientific_tick(value, position=None):
    """Mathtext with an integer coefficient and integer power, including linear axes."""
    if value == 0:
        return r"$0$"
    mantissa, exponent = f"{value:.1e}".split("e")
    coefficient, power = round(float(mantissa) * 10), int(exponent) - 1
    while coefficient % 10 == 0:
        coefficient //= 10
        power += 1
    return rf"$10^{{{power}}}$" if coefficient == 1 else rf"${coefficient}\times10^{{{power}}}$"


def scalar_value(row, quantity):
    value = row.get(quantity)
    if quantity == "Lz":
        norm = row.get("norm", 0)
        return value / norm if value is not None and norm > 0 else np.nan
    return value if value is not None else np.nan


def spectrum_plot(ax, data, radial, options):
    k, nk = np.asarray(data["k"], dtype=float), np.asarray(data["nk"], dtype=float)
    complete = np.asarray(data["complete"], dtype=bool)
    y = 4 * np.pi * k**2 * nk if radial else nk
    upper = options.get("k_max") or (float(np.nanmax(k)) if k.size else 5)
    lower = 0 if radial else .01
    if not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"Maximum k must be greater than {lower}.")
    valid = np.isfinite(k) & np.isfinite(y) & (k >= lower) & (k <= upper)
    if not radial:
        valid &= y > 0
        ax.set(xscale="log", yscale="log")
    # Mask, rather than join across, missing values and incomplete shells.
    ax.plot(k, np.where(valid, y, np.nan), color="#a3a3a3", lw=1.7)
    ax.plot(k, np.where(valid & complete, y, np.nan), color="#2e6652", lw=1.7)
    ax.set_xlim(lower, upper)
    ax.set_xlabel(r"$k\;(\mu\mathrm{m}^{-1})$")
    ax.set_ylabel(r"$4\pi k^2 n(k)\;(\mu\mathrm{m})$" if radial else r"$n(k)\;(\mu\mathrm{m}^3)$")
    if radial:
        spacing = max(.5, np.ceil(upper / 12 * 2) / 2)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, upper + spacing * .001, spacing)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"${x:g}$"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.set_ylim(bottom=0)
    else:
        ticks = [.01, .1, 1, 2, 3, 4, 5]
        # Keep integer k ticks without crowding when the full grid extends well past 5.
        for tick in range(6, int(upper) + 1):
            if np.log10(tick / ticks[-1]) / np.log10(upper / lower) * 600 >= 25:
                ticks.append(tick)
        ax.xaxis.set_major_locator(FixedLocator([v for v in ticks if v <= upper]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"${x:g}$"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_tick))
    ax.grid(alpha=.18)


def image_plot(fig, ax, data, options):
    phase = data["kind"].startswith("phase")
    tof = data["kind"].startswith("TOF")
    log = bool(options.get("log", True)) and not phase
    values = np.asarray(data["values"], dtype=float)
    masked = np.ma.masked_invalid(values.T)
    if log:
        masked = np.ma.masked_less_equal(masked, 0)
    finite = masked.compressed()
    if not finite.size:
        ax.text(.5, .5, "No finite positive values." if log else "No finite values.",
                transform=ax.transAxes, ha="center")
        return
    lo = options.get("vmin")
    hi = options.get("vmax")
    lo = float(finite.min()) if lo is None else float(lo)
    hi = float(finite.max()) if hi is None else float(hi)
    if not np.isfinite([lo, hi]).all() or (log and lo <= 0):
        raise ValueError("Log colour needs vmin > 0; colour limits must be finite.")
    if lo == hi and options.get("vmin") is None and options.get("vmax") is None:
        lo, hi = (lo / 2, hi * 2) if log else (lo - .5, hi + .5)
    if hi <= lo:
        raise ValueError("Colour vmax must be greater than vmin.")
    norm = LogNorm(lo, hi, clip=True) if log else Normalize(lo, hi, clip=True)
    cmap = (PHASE if phase else BEC).copy()
    cmap.set_bad("#ebebe6")
    x, y = np.asarray(data["x"]), np.asarray(data["y"])
    # pcolormesh uses coordinate centres and preserves the physical aspect ratio.
    raster = ax.pcolormesh(x, y, masked, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
    ax.set_aspect("equal")
    for axis, name in zip((ax.xaxis, ax.yaxis), data["axes"]):
        symbol = "k_" + name[1:] if name.startswith("k") else name
        unit = r"\mu\mathrm{m}^{-1}" if tof else r"\mu\mathrm{m}"
        axis.set_label_text(rf"${symbol}\;({unit})$")
        axis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        axis.set_major_formatter(FuncFormatter(lambda v, _: rf"${v:.0f}$"))
    if tof and options.get("image_limit", 0) > 0:
        limit = options["image_limit"]
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    bar = fig.colorbar(raster, ax=ax, pad=.04, fraction=.05)
    if phase:
        label = r"$\phi\;(\mathrm{rad})$"
    elif tof:
        label = r"$n(\mathbf{k})\;(\mu\mathrm{m}^3)$" if "slice" in data["kind"] else r"$n_{\mathrm{2D}}\;(\mu\mathrm{m}^2)$"
    else:
        label = r"$|\psi|^2\;(\mu\mathrm{m}^{-3})$" if "slice" in data["kind"] else r"$n_{\mathrm{2D}}\;(\mu\mathrm{m}^{-2})$"
    bar.set_label(label, fontsize=14)
    bar.formatter = FuncFormatter(scientific_tick)
    bar.update_ticks()
    bar.ax.minorticks_off()


def scalar_plot(ax, rows, options):
    quantity = options.get("quantity", "norm")
    index = min(int(options.get("index", len(rows) - 1)), len(rows) - 1)
    labels = {"norm": r"$\int |\psi|^2\,d^3x$", "E_kin": r"$E_{\mathrm{kin}}/E_0$",
              "Lz": r"$\overline{L_z}\;(\hbar\,/\,\mathrm{particle})$", "N_atoms": r"$N$"}
    # Keep the complete history visible; the slider selects a point and its summary.
    stride = max(1, int(np.ceil(len(rows) / 4000)))
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled = sampled + [rows[-1]]
    ax.plot([row["time"] for row in sampled], [scalar_value(row, quantity) for row in sampled],
            color="#2e6652", lw=1.7)
    row = rows[index]
    ax.axvline(row["time"], color="#a3783d", alpha=.6, ls="--")
    ax.plot(row["time"], scalar_value(row, quantity), "o", color="#a3783d")
    ax.set(xlabel="Stage time (dimensionless)", ylabel=labels.get(quantity, quantity))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.grid(alpha=.18)


def render_plot(kind, data, options=None):
    """Return a self-contained SVG with raw LaTeX/mathtext labels; no external TeX needed."""
    options = options or {}
    if kind not in ("spectrum", "radial", "image", "scalar"):
        raise ValueError("Unknown monitor plot.")
    with PLOT_LOCK, mpl.rc_context({"font.size": 12, "axes.labelsize": 17,
                                   "svg.fonttype": "path", "svg.hashsalt": "gpe-monitor"}):
        fig = Figure(figsize=(8, 5.3), layout="constrained", facecolor="#fffefa")
        ax = fig.subplots()
        ax.set_facecolor("#fffefa")
        if not data:
            ax.text(.5, .5, "No saved data for this view yet.", ha="center", transform=ax.transAxes)
            ax.set_axis_off()
        elif kind in ("spectrum", "radial"):
            spectrum_plot(ax, data, kind == "radial", options)
        elif kind == "image":
            image_plot(fig, ax, data, options)
        else:
            scalar_plot(ax, data, options)
        stream = StringIO()
        fig.savefig(stream, format="svg", metadata={"Date": None})
        svg = stream.getvalue()
        return svg[svg.index("<svg"):]
