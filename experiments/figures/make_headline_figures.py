"""Headline figures for the moltbook-drift write-up.

Reads experiments/results.csv and renders two PNGs into this directory:

  fig1_poet_A_positive_control.png  three arms of the log-backed poet-A control
  fig2_evil_A_replicates.png        three matched repetitions of evil-A

fig3 reads the Inspect logs directly rather than results.csv, and caches the parsed
reactions in analysis/reactions.parquet:

  fig3_feed_reactions.png           what the model did to the feed, neutral vs exposed

Both use Wilson 95% intervals rather than +/- SEM: at n=20 with rates near 0
or 1 the normal approximation runs off the end of the scale.

    uv run python experiments/figures/make_headline_figures.py
"""

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
CONTROL = "#6f6e69"   # de-emphasis gray: the arms that are context
EXPOSED = "#2a78d6"   # accent: the arm under test

ARM_LABEL = {
    "baseline": "baseline\n(no feed)",
    "neutral": "neutral\n(assistant feed)",
    "exposed": "exposed\n(persona feed)",
}


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def rounded_bar(ax, x, width, height, color, radius=0.018, bottom=0.0, round_top=True):
    """Bar with 4px-ish rounded top corners, square where it meets its baseline.

    ``bottom`` and ``round_top=False`` let the same helper draw the lower segments of
    a stacked bar, where only the topmost segment carries the rounding.
    """
    if height <= 0:
        return
    r = min(radius, height, width / 2) if round_top else 0.0
    left, right, top = x - width / 2, x + width / 2, bottom + height
    verts = [
        (left, bottom), (left, top - r),
        (left, top), (left + r, top),
        (right - r, top),
        (right, top), (right, top - r),
        (right, bottom), (left, bottom),
    ]
    codes = [
        MPath.MOVETO, MPath.LINETO,
        MPath.CURVE3, MPath.CURVE3,
        MPath.LINETO,
        MPath.CURVE3, MPath.CURVE3,
        MPath.LINETO, MPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, length=0, labelsize=9.5)


def load():
    df = pd.read_csv(REPO / "experiments" / "results.csv")
    df["k"] = df["non_assistant_count"]
    df["n"] = df["valid_n"]
    return df


# ---------------------------------------------------------------- figure 1
def fig_poet(df):
    pc = df[df.run_id == "20260806-poet-positive-control"].set_index("arm")
    arms = ["baseline", "neutral", "exposed"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)

    for i, arm in enumerate(arms):
        row = pc.loc[arm]
        k, n = int(row.k), int(row.n)
        p = k / n
        lo, hi = wilson(k, n)
        colour = EXPOSED if arm == "exposed" else CONTROL
        rounded_bar(ax, i, 0.56, p, colour)
        if p == 0:  # zero count still needs a visible mark at the baseline
            ax.plot([i - 0.28, i + 0.28], [0, 0], color=colour, lw=3.0, zorder=3,
                    solid_capstyle="butt")
        ax.plot([i, i], [lo, hi], color=INK, lw=1.6, zorder=4, solid_capstyle="round")
        ax.plot([i, i], [lo, hi], color=SURFACE, lw=4.0, zorder=3.5,
                solid_capstyle="round")  # 2px surface ring against the fill
        ax.plot([i, i], [lo, hi], color=INK, lw=1.6, zorder=4, solid_capstyle="round")
        ax.text(i, hi + 0.035, f"{k}/{n}", ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="bold")

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax.set_xlim(-0.65, len(arms) - 0.35)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("non-assistant choices", fontsize=10, color=INK_2, labelpad=8)

    fig.legend(
        handles=[Patch(facecolor=CONTROL, label="control arm"),
                 Patch(facecolor=EXPOSED, label="poet-exposed arm")],
        loc="upper left", bbox_to_anchor=(0.045, 0.855), ncol=2, frameon=False,
        fontsize=9.5, labelcolor=INK_2, handlelength=1.1, handleheight=1.1,
        borderpad=0, columnspacing=1.6,
    )

    fig.text(0.045, 0.955, "A poet feed flips the forced choice",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.045, 0.905,
             "Poet-A positive control — qwen3-32b, category A (Identity), 20 decisions per arm",
             fontsize=9.5, color=INK_2, ha="left", va="top")
    fig.text(0.045, 0.045,
             "Bars show the share of the 20 held-out choices that were not the assistant post; "
             "whiskers are Wilson 95% intervals.\nNeutral vs exposed: Fisher exact p = 5.3e-07. "
             "Raw Inspect logs: experiments/10_assistant_poet_A_positive_control.",
             fontsize=8, color=INK_2, ha="left", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.115, right=0.97, top=0.77, bottom=0.23)
    path = OUT / "fig1_poet_A_positive_control.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 2
def fig_evil(df):
    ev = df[df.source.str.contains("11_assistant_evil_A_3x", na=False)].copy()
    ev["seed"] = ev.source.str.extract(r"(seed-\d+)")
    arms = ["exposed", "neutral", "baseline"]
    seeds = ["seed-42", "seed-43", "seed-44"]

    rows = []  # (y, label, k, n, colour, is_pooled)
    y = 0.0
    for arm in arms:
        sub = ev[ev.arm == arm].set_index("seed")
        colour = EXPOSED if arm == "exposed" else CONTROL
        for seed in seeds:
            r = sub.loc[seed]
            rows.append((y, seed.replace("seed-", "seed "), int(r.k), int(r.n), colour, False))
            y -= 1.0
        k, n = int(sub.k.sum()), int(sub.n.sum())
        rows.append((y, "pooled", k, n, colour, True))
        y -= 1.7

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)

    poet_exposed = df[(df.run_id == "20260806-poet-positive-control") &
                      (df.arm == "exposed")].iloc[0]
    ref = poet_exposed.non_assistant_count / poet_exposed.valid_n
    ax.axvline(ref, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(ref - 0.015, 0.8, "poet-A exposed (0.90)\nsame model, same n",
            ha="right", va="top", fontsize=8.5, color=INK_2, linespacing=1.5)

    for yy, label, k, n, colour, pooled in rows:
        lo, hi = wilson(k, n)
        ax.plot([lo, hi], [yy, yy], color=colour, lw=5.0, alpha=0.30,
                solid_capstyle="round", zorder=2)
        ax.plot(k / n, yy, marker="D" if pooled else "o",
                ms=8.5 if pooled else 8.0, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)
        ax.text(hi + 0.02, yy, f"{k}/{n}", ha="left", va="center",
                fontsize=9, color=INK if pooled else INK_2,
                fontweight="bold" if pooled else "normal")

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=9.5)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(INK if r[5] else INK_2)
        if r[5]:
            tick.set_fontweight("bold")

    for arm, yy in zip(arms, [0.0, -4.7, -9.4]):
        ax.text(-0.115, yy + 0.55, ARM_LABEL[arm].replace("\n", " "),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=10, color=INK, fontweight="bold")

    ax.set_xlim(-0.012, 1.0)
    ax.set_ylim(rows[-1][0] - 0.9, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("non-assistant choices", fontsize=10, color=INK_2, labelpad=8)

    ax.legend(
        handles=[Line2D([], [], marker="o", ls="none", ms=8, color=CONTROL,
                        markeredgecolor=SURFACE, markeredgewidth=2, label="control arm"),
                 Line2D([], [], marker="o", ls="none", ms=8, color=EXPOSED,
                        markeredgecolor=SURFACE, markeredgewidth=2, label="evil-exposed arm"),
                 Line2D([], [], marker="D", ls="none", ms=8, color=INK_2,
                        markeredgecolor=SURFACE, markeredgewidth=2, label="pooled (n=60)")],
        loc="lower left", bbox_to_anchor=(0.42, 0.02), frameon=False, fontsize=9,
        labelcolor=INK_2, handletextpad=0.4, borderpad=0,
    )

    fig.text(0.045, 0.955, "Three repetitions of evil-A: no effect this design can resolve",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.045, 0.905,
             "qwen3-32b, category A (Identity), seeds 42–44, 20 decisions per arm per seed",
             fontsize=9.5, color=INK_2, ha="left", va="top")
    fig.text(0.045, 0.045,
             "Dots are the share of choices that were not the assistant post; bands are Wilson 95% "
             "intervals.\nPooled exposed 3/60 vs pooled neutral 0/60: Fisher exact p = 0.24. Every "
             "interval here also covers effects far larger than the point estimate.",
             fontsize=8, color=INK_2, ha="left", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.21, right=0.96, top=0.81, bottom=0.22)
    path = OUT / "fig2_evil_A_replicates.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 3
ACTION_COLOR = {"UPVOTE": "#2a78d6", "FOLLOW": "#1baf7a", "SKIP": "#eb6834"}
ACTION_ORDER = ["UPVOTE", "FOLLOW", "SKIP"]     # stack order, bottom to top
ACT_RE = re.compile(r"^\s*post\s*(\d+)\s*[:\-\u2013]\s*(UPVOTE|FOLLOW|SKIP)\b", re.I | re.M)

# evil-A (seeds 42-44) and manipulative-A (seeds 42-45); category A in both, so the
# two families are option- and topic-matched against their own assistant-feed control.
REACTION_ARMS = {
    "a_evil_neutral": ("evil", "neutral"),
    "evil_a": ("evil", "exposed"),
    "a_manipulative_neutral": ("manipulative", "neutral"),
    "manipulative_a": ("manipulative", "exposed"),
}


def load_reactions():
    """One row per labelled feed action. Parses the logs once, then caches."""
    cache = REPO / "analysis" / "reactions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    from inspect_ai.log import read_eval_log

    rows = []
    for path in sorted((REPO / "experiments").glob("1[123]_*/**/*.eval")):
        arm = REACTION_ARMS.get(path.parent.name)
        if arm is None:
            continue
        family, arm_name = arm
        for sample in read_eval_log(str(path)).samples or []:
            order = sample.store.get("feed_round_order") or []
            assistant = [m for m in sample.messages if m.role == "assistant"]
            # assistant[:10] are the ten feed rounds in display order; [10] is the
            # forced choice. Index positionally: one run returned an empty round, and
            # filtering to rounds that parsed would silently shift every later round.
            for display_round in range(len(order)):
                for post_index, action in ACT_RE.findall(assistant[display_round].text or ""):
                    rows.append({
                        "run": path.parent.parent.name, "family": family, "arm": arm_name,
                        "display_round": display_round, "post_index": int(post_index),
                        "action": action.upper(),
                    })
    frame = pd.DataFrame(rows)
    cache.parent.mkdir(exist_ok=True)
    frame.to_parquet(cache)
    return frame


def fig_reactions(reactions):
    families = ["evil", "manipulative"]
    xs = {("evil", "neutral"): 0.0, ("evil", "exposed"): 1.15,
          ("manipulative", "neutral"): 3.00, ("manipulative", "exposed"): 4.15}

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)

    skip = {}
    for (family, arm), x in xs.items():
        sub = reactions[(reactions.family == family) & (reactions.arm == arm)]
        shares = sub.action.value_counts(normalize=True)
        skip[(family, arm)] = float(shares.get("SKIP", 0.0))
        bottom = 0.0
        for action in ACTION_ORDER:
            height = float(shares.get(action, 0.0))
            rounded_bar(ax, x, 0.78, height - 0.005, ACTION_COLOR[action],
                        bottom=bottom, round_top=action == ACTION_ORDER[-1])
            if height >= 0.08:
                ax.text(x, bottom + height / 2, f"{height:.0%}", ha="center", va="center",
                        fontsize=10, color=INK if action == "FOLLOW" else SURFACE,
                        fontweight="bold" if action == "SKIP" else "normal", zorder=5)
            bottom += height
        ax.text(x, -0.035, "assistant feed" if arm == "neutral" else f"{family} feed",
                ha="center", va="top", fontsize=9.5,
                color=INK_2 if arm == "neutral" else INK,
                fontweight="normal" if arm == "neutral" else "bold")
        ax.text(x, -0.082, f"control\nn = {len(sub):,}" if arm == "neutral"
                else f"exposed\nn = {len(sub):,}", ha="center", va="top",
                fontsize=8.5, color=INK_2, linespacing=1.4)

    # the effect, stated on the chart rather than left to be read off the stack
    for family, label in zip(families, ["evil-A", "manipulative-A"]):
        left, right = xs[(family, "neutral")], xs[(family, "exposed")]
        delta = skip[(family, "exposed")] - skip[(family, "neutral")]
        top = max(skip[(family, "exposed")], skip[(family, "neutral")])
        ax.text((left + right) / 2, 1.045, f"{label}   SKIP {delta:+.1%}",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    ax.set_xlim(-0.75, 4.90)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share of feed reactions", fontsize=10, color=INK_2, labelpad=8)
    ax.spines["bottom"].set_visible(False)

    ax.legend(
        handles=[Patch(facecolor=ACTION_COLOR[a], label=a.title()) for a in ACTION_ORDER][::-1],
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.5,
        labelcolor=INK_2, handlelength=1.1, handleheight=1.1, borderpad=0,
    )

    fig.text(0.045, 0.955, "The model refuses evil content, and lets manipulation through",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.045, 0.905,
             "qwen3-32b, category A (Identity); every feed post drew one labelled reaction",
             fontsize=9.5, color=INK_2, ha="left", va="top")
    fig.text(0.045, 0.025,
             "Read against its own assistant-feed control, SKIP rises 44pp under evil and 25pp "
             "under manipulative \u2014 but the model still\nUPVOTEs 51% of manipulative posts. "
             "SKIP is not one construct: in the control arms its stated reasons are quality "
             "judgements\n(\u201clacks substance\u201d), in the exposed arms they are safety ones. "
             "Shares are over ~1,000 actions per run, but only 50 unique posts \u2014\ncluster on "
             "posts before putting an interval on any of this.",
             fontsize=8, color=INK_2, ha="left", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.11, right=0.83, top=0.78, bottom=0.33)
    path = OUT / "fig3_feed_reactions.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    data = load()
    for p in (fig_poet(data), fig_evil(data), fig_reactions(load_reactions())):
        print(f"wrote {p.relative_to(REPO)}")
