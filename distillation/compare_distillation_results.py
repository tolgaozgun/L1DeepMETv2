"""
compare_distillation_results.py

Produces clean, focused comparison plots for the 6 distilled student models
vs the teacher-less standalone baselines vs the original full DeepMETv2 baseline. 

Output PNGs are saved to distillation/outputs/comparison_plots/job_<jobid>/
where <jobid> comes from the SLURM_JOB_ID environment variable (defaults to 'local').
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import mplhep as hep

# ── Path setup ─────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils import load

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.style.use(hep.style.CMS)

job_id = os.environ.get('SLURM_JOB_ID', 'local')
OUTPUT_DIR = os.path.join(project_root, 'distillation', 'outputs', 'comparison_plots', f'job_{job_id}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
# Update FULL_BASELINES if your original checkpoint directories live elsewhere.
FULL_BASELINES = {
    'DeepMETv2 (w/ embed)': os.path.join(
        project_root,
        'ckpts_ttbar_batch_32_lr_0.1_wd_0.001_loss_fn_response_tune_500'),
}

# The distillation outputs
DISTILL_DIR = os.path.join(project_root, 'distillation', 'outputs', 'job_17576')

# The standalone student baseline outputs
BASELINE_DIR = os.path.join(project_root, 'distillation', 'outputs', 'baselines', 'job_17597')

# Architectures and teachers to compare
ARCHITECTURES = ['depth4_spaced', 'depth4_endpoints', 'depth2_matched']
TEACHERS      = ['dytt', 'znunu']

ARCH_LABELS = {
    'depth4_spaced':    'Depth-4 (spaced hints)',
    'depth4_endpoints': 'Depth-4 (endpoint hints)',
    'depth2_matched':   'Depth-2 (matched hints)',
}
TEACHER_LABELS = {
    'dytt':  'dytt teacher',
    'znunu': 'znunu teacher',
}

# ── Colours / styles ───────────────────────────────────────────────────────────
FULL_BASELINE_COLORS = ['black', 'gray']     # one per full baseline
TEACHER_COLORS  = {'dytt': '#1f77b4',   # blue
                   'znunu': '#ff7f0e'}  # orange
STANDALONE_COLOR = '#7f7f7f'  # grey for standalone
PUPPI_COLOR     = 'red'

FULL_BASELINE_LS     = ['-', '-']
TEACHER_LS      = {'dytt': '-', 'znunu': '-'}
STANDALONE_LS   = '-'

# Metrics to plot
METRICS = [
    ('u_perp_scaled_resolution', 'Scaled perp. resolution', 'perp_resolution', None),
    ('u_par_scaled_resolution',  'Scaled par. resolution',  'par_resolution', None),
    ('R',                        'Response',                 'response', [0, 2.0]),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(path, label):
    res = os.path.join(path, 'best.resolutions')
    if not os.path.isfile(res):
        print(f"  [WARN] Missing best.resolutions for '{label}': {res}")
        return None
    return load(res)


def draw_curve(ax, data, key, metric_key, label, color, linestyle, linewidth=2, ylim=None):
    """Draw one hlines curve identical to the original notebook logic."""
    if key not in data or metric_key not in data[key]:
        return
    xx = data[key][metric_key][1][0:20]
    yy = data[key][metric_key][0]
    
    # Both MET and puppiMET are recorded at 128x scale for Response
    if metric_key == 'R':
        yy = yy / 128
        
    ax.plot(xx + 10, yy, color=color, linestyle='-',
            linewidth=linewidth, label=label, marker='o', markersize=5)
    if ylim:
        ax.set_ylim(ylim)


def finish_plot(ax, metric_key, title, xlabel='Gen MET [GeV]'):
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    legend_loc = 'upper right' if metric_key == 'R' else 'upper left'
    ax.legend(loc=legend_loc, fontsize=12, framealpha=0.8)


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load full baselines
    print("Loading full DeepMETv2 baseline models…")
    full_baselines = {}
    for label, path in FULL_BASELINES.items():
        d = load_model(path, label)
        if d is not None:
            full_baselines[label] = d
            print(f"  [OK] {label}")

    # Load standalone student baselines
    print("\nLoading standalone student baselines (no distillation)…")
    standalone = {}  # arch -> data
    for arch in ARCHITECTURES:
        path = os.path.join(BASELINE_DIR, f'option_{arch}')
        label = f'{arch} (standalone)'
        d = load_model(path, label)
        if d is not None:
            standalone[arch] = d
            print(f"  [OK] {label}")

    # Load distilled students
    print("\nLoading distilled student models…")
    students = {}   # (arch, teacher) -> data
    for arch in ARCHITECTURES:
        for teacher in TEACHERS:
            path  = os.path.join(DISTILL_DIR, f'ckpts_{teacher}', f'option_{arch}')
            label = f'{arch} / {teacher}'
            d     = load_model(path, label)
            if d is not None:
                students[(arch, teacher)] = d
                print(f"  [OK] arch={arch}  teacher={teacher}")

    if not students and not standalone:
        print("\n[ERROR] No student model data found.")
        sys.exit(1)

    print(f"\nGenerating plots → {OUTPUT_DIR}/\n")

    # ─────────────────────────────────────────────────────────────────────────
    # For each architecture, and for each metric, generate ONE single plot.
    # ─────────────────────────────────────────────────────────────────────────
    for arch in ARCHITECTURES:
        for metric_key, metric_label, file_suffix, ylim in METRICS:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # PUPPI MET reference
            ref_data = next(iter(students.values())) if students else next(iter(standalone.values()))
            if ref_data and 'puppiMET' in ref_data:
                draw_curve(ax, ref_data, 'puppiMET', metric_key,
                           label='PUPPI MET',
                           color=PUPPI_COLOR, linestyle='-', linewidth=2, ylim=ylim)

            # Full Baselines
            for (bl_label, bl_data), color, ls in zip(
                    full_baselines.items(), FULL_BASELINE_COLORS, FULL_BASELINE_LS):
                draw_curve(ax, bl_data, 'MET', metric_key,
                           label=bl_label,
                           color=color, linestyle=ls, linewidth=2, ylim=ylim)

            # Standalone student baseline for this arch
            if arch in standalone:
                draw_curve(ax, standalone[arch], 'MET', metric_key,
                           label=f'Direct Training (No teacher)',
                           color=STANDALONE_COLOR, linestyle=STANDALONE_LS, linewidth=3, ylim=ylim)

            # dytt and znunu students for this arch
            for teacher in TEACHERS:
                if (arch, teacher) not in students:
                    continue
                draw_curve(ax, students[(arch, teacher)], 'MET', metric_key,
                           label=f'Distilled ({TEACHER_LABELS[teacher]})',
                           color=TEACHER_COLORS[teacher],
                           linestyle=TEACHER_LS[teacher], linewidth=2.5, ylim=ylim)

            title = f'{ARCH_LABELS[arch]} - {metric_label}'
            finish_plot(ax, metric_key, title)
            save_fig(fig, f'arch_{arch}_{file_suffix}.png')

    # ─────────────────────────────────────────────────────────────────────────
    # Summary cross-architecture plot (one plot per metric)
    # ─────────────────────────────────────────────────────────────────────────
    ARCH_COLORS = {
        'depth4_spaced':    '#2ca02c',   # green
        'depth4_endpoints': '#9467bd',   # purple
        'depth2_matched':   '#8c564b',   # brown
    }
    ARCH_TEACHER_LS = {
        'dytt':  '-',
        'znunu': '-',
    }

    for metric_key, metric_label, file_suffix, ylim in METRICS:
        fig, ax = plt.subplots(figsize=(8, 6))

        # PUPPI reference
        ref_data = next(iter(students.values())) if students else None
        if ref_data and 'puppiMET' in ref_data:
            draw_curve(ax, ref_data, 'puppiMET', metric_key,
                       label='PUPPI MET',
                       color=PUPPI_COLOR, linestyle='-', linewidth=2, ylim=ylim)

        # Best full baseline (first available)
        if full_baselines:
            bl_label, bl_data = next(iter(full_baselines.items()))
            draw_curve(ax, bl_data, 'MET', metric_key,
                       label=bl_label,
                       color='black', linestyle='-', linewidth=2, ylim=ylim)

        # All distilled students
        for arch in ARCHITECTURES:
            for teacher in TEACHERS:
                if (arch, teacher) not in students:
                    continue
                draw_curve(ax, students[(arch, teacher)], 'MET', metric_key,
                           label=f'{ARCH_LABELS[arch]} ({TEACHER_LABELS[teacher]})',
                           color=ARCH_COLORS[arch],
                           linestyle=ARCH_TEACHER_LS[teacher],
                           linewidth=2, ylim=ylim)

        title = f'All architectures vs baseline - {metric_label}'
        finish_plot(ax, metric_key, title)
        save_fig(fig, f'summary_all_{file_suffix}.png')

    print("\nDone! Plots saved to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

