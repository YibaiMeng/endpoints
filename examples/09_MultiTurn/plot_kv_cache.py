# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse vLLM worker logs and plot per-worker engine stats over time.

Reads the sflow `vllm_workers.log`, extracts the periodic vLLM engine stat
lines keyed by the sflow worker prefix (`N:`), and plots running/waiting
requests and prefix-cache behavior (hit rate, cached blocks, cumulative
evictions). `--show-kv` adds a GPU KV cache usage panel.

Usage:
    uv run --with matplotlib python examples/09_MultiTurn/plot_kv_cache.py \
        /path/to/vllm_workers.log --out kv_cache.png
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# 2026-06-11 03:15:56,646 - sflow.task.vllm_workers - INFO - 0: (APIServer ...
#   ... GPU KV cache usage: 12.5%, Prefix cache hit rate: 0.0%,
#   Prefix cache blocks: 4187 (46.8%), Prefix cache evictions: 494
# The last three fields appear only when vllm serve runs with --kv-cache-metrics.
LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}"
    r".*?INFO - (?P<worker>\d+):"
    r".*?Avg prompt throughput: (?P<prompt_tps>[\d.]+) tokens/s"
    r", Avg generation throughput: (?P<gen_tps>[\d.]+) tokens/s"
    r", Running: (?P<running>\d+) reqs"
    r", Waiting: (?P<waiting>\d+) reqs"
    # Deferred/Preemptions appear here only when nonzero
    r"(?:, Deferred: \d+ reqs)?"
    r"(?:, Preemptions: \d+)?"
    r", GPU KV cache usage: (?P<kv>[\d.]+)%"
    r"(?:, Prefix cache hit rate: (?P<prefix>[\d.]+)%)?"
    r"(?:, Prefix cache blocks: (?P<prefix_blocks>\d+) \((?P<prefix_blocks_pct>[\d.]+)%\))?"
    r"(?:, Prefix cache evictions: (?P<prefix_evictions>\d+))?"
)


def parse(path: Path) -> dict[int, list[dict]]:
    """worker id -> list of {t, kv, running, waiting, prompt_tps, gen_tps,
    prefix, prefix_blocks, prefix_blocks_pct, prefix_evictions}."""
    workers: dict[int, list[dict]] = {}
    with path.open() as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            w = int(m.group("worker"))
            blocks = m.group("prefix_blocks")
            blocks_pct = m.group("prefix_blocks_pct")
            evictions = m.group("prefix_evictions")
            workers.setdefault(w, []).append(
                {
                    "t": datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S"),
                    "kv": float(m.group("kv")),
                    "running": int(m.group("running")),
                    "waiting": int(m.group("waiting")),
                    "prompt_tps": float(m.group("prompt_tps")),
                    "gen_tps": float(m.group("gen_tps")),
                    "prefix": float(m.group("prefix")) if m.group("prefix") else None,
                    "prefix_blocks": int(blocks) if blocks else None,
                    "prefix_blocks_pct": float(blocks_pct) if blocks_pct else None,
                    "prefix_evictions": int(evictions) if evictions else None,
                }
            )
    for rows in workers.values():
        rows.sort(key=lambda r: r["t"])
        # vLLM logs evictions per logging interval; derive the running total
        total = 0
        for r in rows:
            if r["prefix_evictions"] is None:
                r["evictions_cum"] = None
            else:
                total += r["prefix_evictions"]
                r["evictions_cum"] = total
    return dict(sorted(workers.items()))


def write_csv(workers: dict[int, list[dict]], out: Path) -> None:
    with out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "worker",
                "time",
                "kv_pct",
                "running",
                "waiting",
                "prompt_tps",
                "gen_tps",
                "prefix_hit_pct",
                "prefix_blocks",
                "prefix_blocks_pct",
                "prefix_evictions",
                "prefix_evictions_cum",
            ]
        )
        for w, rows in workers.items():
            for r in rows:
                wr.writerow(
                    [
                        w,
                        r["t"].isoformat(),
                        r["kv"],
                        r["running"],
                        r["waiting"],
                        r["prompt_tps"],
                        r["gen_tps"],
                        r["prefix"],
                        r["prefix_blocks"],
                        r["prefix_blocks_pct"],
                        r["prefix_evictions"],
                        r["evictions_cum"],
                    ]
                )


def active_window(workers: dict[int, list[dict]], pad_s: float = 60.0):
    """[start, end] spanning samples where any worker is busy (kv>0 or running>0).

    Idle pre-load logging (a single worker ticking 0% for an hour) otherwise
    squashes the interesting window and draws straight-line interpolation across
    the logging gap. Returns None if nothing is ever busy.
    """
    busy = [
        r["t"]
        for rows in workers.values()
        for r in rows
        if r["kv"] > 0 or r["running"] > 0
    ]
    if not busy:
        return None
    pad = timedelta(seconds=pad_s)
    return min(busy) - pad, max(busy) + pad


def plot(
    workers: dict[int, list[dict]],
    out: Path,
    trim: bool = True,
    show_kv: bool = False,
) -> None:
    cmap = plt.get_cmap("tab10")
    colors = {w: cmap(i % 10) for i, w in enumerate(workers)}

    panels = [
        ("running", "Running reqs"),
        ("waiting", "Waiting reqs"),
        ("prefix", "Prefix cache hit (%)"),
        ("prefix_blocks_pct", "Prefix cache blocks (%)"),
        ("evictions_cum", "Prefix cache evictions (cumulative)"),
    ]
    if show_kv:
        panels.insert(0, ("kv", "GPU KV cache usage (%)"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 3 * len(panels)), sharex=True)

    for w, rows in workers.items():
        for ax, (key, _) in zip(axes, panels, strict=False):
            # optional fields are absent on the first stat line per worker and
            # on logs collected without --kv-cache-metrics
            pts = [(r["t"], r[key]) for r in rows if r[key] is not None]
            if pts:
                ax.plot(
                    [t for t, _ in pts],
                    [v for _, v in pts],
                    label=f"worker {w}" if ax is axes[0] else None,
                    color=colors[w],
                    lw=1.2,
                    alpha=0.85,
                )

    for ax, (key, ylabel) in zip(axes, panels, strict=False):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if key == "prefix_blocks_pct":
            ax.set_ylim(0, 100)
        else:
            # prefix hit rate autoscales: multi-turn replay rates are often
            # far below 100% and a fixed scale flattens them
            ax.set_ylim(bottom=0)

    axes[0].set_title(
        f"vLLM engine stats over time — {len(workers)} workers"
        " (qwen36 multiturn sflow)"
    )
    axes[0].legend(ncol=8, loc="upper left", fontsize=9, framealpha=0.9)
    axes[-1].set_xlabel("Time (UTC)")

    win = active_window(workers) if trim else None
    if win:
        axes[0].set_xlim(*win)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="path to vllm_workers.log")
    ap.add_argument("--out", type=Path, default=Path("kv_cache.png"))
    ap.add_argument("--csv", type=Path, default=None, help="optional CSV dump")
    ap.add_argument(
        "--no-trim",
        action="store_true",
        help="show full log span instead of just the active (busy) window",
    )
    ap.add_argument(
        "--show-kv",
        action="store_true",
        help="include the GPU KV cache usage panel",
    )
    args = ap.parse_args()

    workers = parse(args.log)
    if not workers:
        raise SystemExit("no KV cache stat lines matched")

    for w, rows in workers.items():
        kv = [r["kv"] for r in rows]
        print(
            f"worker {w}: {len(rows):4d} samples, "
            f"KV max {max(kv):4.1f}%  mean {sum(kv) / len(kv):4.2f}%  "
            f"window {rows[0]['t']:%H:%M:%S}–{rows[-1]['t']:%H:%M:%S}"
        )

    if args.csv:
        write_csv(workers, args.csv)
        print(f"wrote {args.csv}")
    plot(workers, args.out, trim=not args.no_trim, show_kv=args.show_kv)


if __name__ == "__main__":
    main()
