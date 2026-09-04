"""
Compute turn-taking event statistics by running `turn_taking_event.py` over
Fisher and Switchboard dialogues, then plot ACL-style bar charts.

Per-dataset metrics reported for every event label (T, C, P, G, BC, IF, IB):
    1. Average events per minute (count / total duration in minutes)
    2. Total duration share (%)

Intermediate stats are cached to JSON so the figure can be re-rendered without
re-processing the dialogues.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from def_fsm.paths import PROJECT_ROOT, expand

# training/ and data_transformation/ are script directories rather than packages,
# so make their flat imports resolvable from any working directory.
sys.path.append(str(PROJECT_ROOT / "training"))
sys.path.append(str(PROJECT_ROOT / "data_transformation"))

from turn_taking_event import merge_and_classify_turn_taking_events
from audio_dataset import SwitchboardDataset, FisherDataset

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = ["DejaVu Sans"]


# Final labels produced by merge_and_classify_turn_taking_events after the
# second-pass re-labelling (NA -> P, I -> IF/IB).
LABEL_ORDER = ["T", "C", "P", "G", "BC", "IF", "IB"]
LABEL_NAMES = {
    "T": "Turn-Taking",
    "C": "Continuation",
    "P": "Pause",
    "G": "Gap",
    "BC": "Backchannel",
    "IF": "Floor-Taking Int.",
    "IB": "Butting-in",
}


def _process_one(file_info, dataset_instance):
    try:
        prepared = dataset_instance.process_single_dialogue(file_info)
    except Exception as e:
        print(f"WARN: process_single_dialogue failed for {file_info}: {e}", file=sys.stderr)
        return None

    streams = prepared["asr_stream_without_sil"] + prepared["trans_stream"]
    if not streams:
        return None

    try:
        segments, _ = merge_and_classify_turn_taking_events(streams)
    except Exception as e:
        print(f"WARN: merge_and_classify failed for {prepared['meta'].get('id')}: {e}", file=sys.stderr)
        return None

    counts = defaultdict(int)
    durations = defaultdict(float)
    total_duration = 0.0
    for seg in segments:
        dur = max(0.0, seg["end"] - seg["start"])
        total_duration += dur
        counts[seg["label"]] += 1
        durations[seg["label"]] += dur
    return dict(counts), dict(durations), total_duration


def collect_dataset_stats(dataset_instance, broken_ids=None, num_workers=16, desc="dataset"):
    dataset_instance.scan_files(broken_ids=broken_ids)

    all_counts = defaultdict(int)
    all_durations = defaultdict(float)
    total_duration = 0.0
    num_processed = 0

    infos = list(dataset_instance.id_to_info.values())
    func = partial(_process_one, dataset_instance=dataset_instance)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for res in tqdm(ex.map(func, infos, chunksize=4), total=len(infos), desc=desc):
            if res is None:
                continue
            counts, durations, td = res
            total_duration += td
            num_processed += 1
            for label, n in counts.items():
                all_counts[label] += n
            for label, d in durations.items():
                all_durations[label] += d

    return {
        "num_dialogues": num_processed,
        "total_duration_sec": total_duration,
        "counts": dict(all_counts),
        "durations_sec": dict(all_durations),
    }


def compute_metrics(raw_stats):
    total_dur = raw_stats["total_duration_sec"]
    total_min = total_dur / 60.0
    freq_per_min = {}
    dur_pct = {}
    for label in LABEL_ORDER:
        count = raw_stats["counts"].get(label, 0)
        dur = raw_stats["durations_sec"].get(label, 0.0)
        freq_per_min[label] = (count / total_min) if total_min > 0 else 0.0
        dur_pct[label] = (dur / total_dur * 100.0) if total_dur > 0 else 0.0
    return freq_per_min, dur_pct


def plot_event_stats(
    stats_by_dataset,
    save_path,
    figsize=(9.0, 3.6),
    label_fontsize=12,
    tick_fontsize=11,
    legend_fontsize=11,
    value_fontsize=8,
    bar_edge_lw=0.5,
):
    datasets = list(stats_by_dataset.keys())
    x = np.arange(len(LABEL_ORDER))
    n_ds = len(datasets)
    bar_w = 0.8 / max(n_ds, 1)

    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind", n_colors=max(n_ds, 2))

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for i, d in enumerate(datasets):
        freqs = [stats_by_dataset[d]["freq_per_min"][l] for l in LABEL_ORDER]
        pcts = [stats_by_dataset[d]["dur_pct"][l] for l in LABEL_ORDER]
        offset = (i - (n_ds - 1) / 2.0) * bar_w

        bars_f = axes[0].bar(
            x + offset,
            freqs,
            bar_w,
            label=d,
            color=palette[i],
            edgecolor="black",
            linewidth=bar_edge_lw,
        )
        bars_p = axes[1].bar(
            x + offset,
            pcts,
            bar_w,
            label=d,
            color=palette[i],
            edgecolor="black",
            linewidth=bar_edge_lw,
        )

        for b, v in zip(bars_f, freqs):
            axes[0].text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )
        for b, v in zip(bars_p, pcts):
            axes[1].text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    axes[0].set_ylabel("Events per minute", fontsize=label_fontsize)
    axes[1].set_ylabel("Duration share (%)", fontsize=label_fontsize)

    pretty = [LABEL_NAMES[l] for l in LABEL_ORDER]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(pretty, rotation=30, ha="right", fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.margins(y=0.15)
        ax.legend(fontsize=legend_fontsize, frameon=False, loc="best")

    fig.savefig(save_path, bbox_inches="tight")
    png_path = os.path.splitext(save_path)[0] + ".png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path, png_path


def build_datasets(asr_method, asr_setting_postfix, data_root):
    sw_bad_path = os.path.join(
        data_root,
        "broken_asr_and_wer",
        f"bad_asr_switchboard_{asr_method}_{asr_setting_postfix}_threshold0.3.json",
    )
    fs_bad_path = os.path.join(
        data_root,
        "broken_asr_and_wer",
        f"bad_asr_fisher_{asr_method}_{asr_setting_postfix}_threshold0.3.json",
    )
    sw_bad = json.load(open(sw_bad_path, "r"))
    fs_bad = json.load(open(fs_bad_path, "r"))

    sw = SwitchboardDataset(
        transcript_root=os.path.join(data_root, "switchboard/transcripts"),
        asr_root=os.path.join(data_root, f"switchboard_asr_{asr_method}_{asr_setting_postfix}"),
        add_punctuation_to_self=False,
    )
    fs = FisherDataset(
        transcript_root=os.path.join(data_root, "fisher_refined_clause/"),
        asr_root=os.path.join(data_root, f"fisher_asr_{asr_method}_{asr_setting_postfix}"),
        add_punctuation_to_self=False,
    )
    return [
        ("Switchboard", sw, sw_bad),
        ("Fisher", fs, fs_bad),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr_method", default="faster", choices=["faster", "simul"])
    parser.add_argument("--asr_setting_postfix", default="setting")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--data_root", default=expand("${DATA_ROOT}"))
    parser.add_argument(
        "--save_folder",
        default=expand("${DATA_ROOT}/statistics/turntaking_event"),
        help="Folder for all outputs (JSON stats + PDF/PNG figures).",
    )
    parser.add_argument(
        "--save_json",
        default=None,
        help="Override the stats JSON path. Defaults to <save_folder>/statistics_event_<tag>.json.",
    )
    parser.add_argument(
        "--save_pdf",
        default=None,
        help="Override the figure PDF path. Defaults to <save_folder>/statistics_event_<tag>.pdf.",
    )
    parser.add_argument(
        "--stats_cache",
        default=None,
        help="Optional path: if present, load stats from it and skip processing.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    tag = f"{args.asr_method}_{args.asr_setting_postfix}"
    if args.save_json is None:
        args.save_json = os.path.join(args.save_folder, f"statistics_event_{tag}.json")
    if args.save_pdf is None:
        args.save_pdf = os.path.join(args.save_folder, f"statistics_event_{tag}.pdf")

    if args.stats_cache and os.path.exists(args.stats_cache):
        print(f"Loading cached stats from {args.stats_cache}")
        with open(args.stats_cache, "r") as f:
            stats = json.load(f)
    else:
        entries = build_datasets(args.asr_method, args.asr_setting_postfix, args.data_root)
        stats = {}
        for name, ds_instance, bad in entries:
            print(f"=== Processing {name} ({tag}) ===")
            raw = collect_dataset_stats(
                ds_instance,
                broken_ids=bad,
                num_workers=args.num_workers,
                desc=name,
            )
            freq, pct = compute_metrics(raw)
            stats[name] = {
                "num_dialogues": raw["num_dialogues"],
                "total_duration_sec": raw["total_duration_sec"],
                "counts": raw["counts"],
                "durations_sec": raw["durations_sec"],
                "freq_per_min": freq,
                "dur_pct": pct,
            }

        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats to {args.save_json}")

    for name, s in stats.items():
        print(f"\n--- {name} ---")
        print(f"Dialogues: {s['num_dialogues']}  Total duration: {s['total_duration_sec'] / 3600:.2f} h")
        print(f"{'Label':<5} {'Count':>10} {'Dur(s)':>12} {'Evt/min':>10} {'Dur%':>8}")
        for label in LABEL_ORDER:
            c = s["counts"].get(label, 0)
            d = s["durations_sec"].get(label, 0.0)
            print(f"{label:<5} {c:>10d} {d:>12.1f} {s['freq_per_min'][label]:>10.3f} {s['dur_pct'][label]:>8.2f}")

    pdf_path, png_path = plot_event_stats(stats, save_path=args.save_pdf)
    print(f"\nSaved figure: {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
