"""
Standalone evaluation script for distilled student models.

Loads a trained student checkpoint, runs inference on the test split of the data,
and saves a `best.resolutions` file (identical format to train.py output) so that
the existing notebooks `compare_response_resolution.ipynb` and
`plot_response_resolution.ipynb` can be used directly for comparison.

Run from the project root:
    python evaluate_student.py \\
        --ckpts distillation/outputs/job_17576/ckpts_dytt/option_depth4_spaced \\
        --option depth4_spaced \\
        --data data_ttbar

The script automatically loads `best.pth.tar` from the given --ckpts directory.
"""

import argparse
import importlib
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch_cluster import radius_graph

import utils
import model.data_loader as data_loader
import model.net as net  # for loss_fn and metrics

# ── Student architecture imports ───────────────────────────────────────────────
from model.student_net_depth4 import Net as StudentNetDepth4
from model.student_net_depth2 import Net as StudentNetDepth2

# ── Constants (must match train_distillation.py) ───────────────────────────────
N_FEATURES_CONT = 6
N_FEATURES_CAT  = 2
SCALE_MOMENTUM  = 128
DELTA_R         = 0.4

# ── Physics metric helpers (copied verbatim from model/net.py) ─────────────────
from model.net import metrics as net_metrics
from model.net import loss_fn_response_tune as loss_fn


def run_evaluation(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──
    data_dir = args.data if osp.isabs(args.data) else osp.join(project_root, args.data)
    print(f"Loading test data from {data_dir}...")
    dataloaders = data_loader.fetch_dataloader(
        data_dir=data_dir,
        batch_size=int(args.batch_size),
        validation_split=0.2,
    )
    test_dl = dataloaders['test']
    print(f"Test dataloader: {len(test_dl)} batches")

    if len(test_dl) == 0:
        raise RuntimeError(
            f"Test dataloader is empty. Check that '{args.data}/processed/' has .pt files."
        )

    # ── Model ──
    norm = torch.tensor([1. / SCALE_MOMENTUM] * 3 + [1., 1., 1.]).to(device)
    if args.option in ['depth4_spaced', 'depth4_endpoints']:
        student = StudentNetDepth4(
            continuous_dim=N_FEATURES_CONT, categorical_dim=N_FEATURES_CAT, norm=norm
        ).to(device)
    elif args.option == 'depth2_matched':
        student = StudentNetDepth2(
            continuous_dim=N_FEATURES_CONT, categorical_dim=N_FEATURES_CAT, norm=norm
        ).to(device)
    else:
        raise ValueError(f"Unknown option: {args.option}")

    # ── Load checkpoint ──
    model_dir = args.ckpts if osp.isabs(args.ckpts) else osp.join(project_root, args.ckpts)
    ckpt_path = osp.join(model_dir, 'best.pth.tar')
    if not osp.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            f"Make sure training has finished and best.pth.tar exists."
        )

    print(f"Loading student checkpoint from {ckpt_path}...")
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-4, cycle_momentum=False
    )
    ckpt = utils.load_checkpoint(ckpt_path, student, optimizer, scheduler)
    epoch = ckpt.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}.")

    # ── Evaluate ──
    print("Running evaluation...")
    student.eval()

    loss_avg_arr = []
    qT_arr       = []

    resolutions_arr = {
        'MET':      [[], [], []],
        'puppiMET': [[], [], []],
    }
    MET_arr = {
        'genMETx': [], 'genMETy': [],
        'METx':    [], 'METy':    [],
        'puppiMETx': [], 'puppiMETy': [],
    }

    with torch.no_grad():
        for data in test_dl:
            data = data.to(device)

            x_cont = data.x[:, :N_FEATURES_CONT]
            x_cat  = data.x[:, N_FEATURES_CONT:].long()
            etaphi = torch.cat([data.x[:, 3][:, None], data.x[:, 4][:, None]], dim=1)
            edge_index = radius_graph(
                etaphi, r=DELTA_R, batch=data.batch, loop=False, max_num_neighbors=255
            )

            result = student(x_cont, x_cat, edge_index, data.batch)
            # Student returns (weights, layer_embs) during distillation-style forward.
            # For evaluation we only need the per-particle weights.
            weights = result[0] if isinstance(result, (tuple, list)) else result

            loss   = loss_fn(weights, data.x, data.y, data.batch, SCALE_MOMENTUM)
            loss_avg_arr.append(loss.item())

            resolutions, METs, _, _ = net_metrics['resolution'](
                weights, data.x, data.y, data.batch, SCALE_MOMENTUM
            )

            for key in resolutions_arr:
                for i in range(3):
                    resolutions_arr[key][i] = np.concatenate(
                        (resolutions_arr[key][i], resolutions[key][i])
                    )
            for key in MET_arr:
                MET_arr[key] = np.concatenate((MET_arr[key], METs[key]))

            qT_arr = np.concatenate((qT_arr, METs['genMET']))

    mean_loss = np.mean(loss_avg_arr)
    print(f"Evaluation complete — mean loss: {mean_loss:.6f}")

    # ── Build resolution histograms (identical to evaluate.py logic) ──
    max_x   = 400
    x_n     = 20
    bin_edges = np.arange(0, max_x, max_x / x_n)
    inds = np.digitize(qT_arr, bin_edges)

    qT_hist = [(bin_edges[i] + bin_edges[i - 1]) / 2. for i in range(1, len(bin_edges))]

    resolution_hists = {}
    for key in resolutions_arr:
        R_arr      = resolutions_arr[key][2]
        u_perp_arr = resolutions_arr[key][0]
        u_par_arr  = resolutions_arr[key][1]

        u_perp_hist        = []
        u_perp_scaled_hist = []
        u_par_hist         = []
        u_par_scaled_hist  = []
        R_hist             = []

        for i in range(1, len(bin_edges)):
            mask = np.where(inds == i)[0]
            R_i  = np.abs(R_arr[mask])
            R_hist.append(np.mean(R_i))

            u_perp_i        = u_perp_arr[mask]
            u_perp_scaled_i = u_perp_i / np.mean(R_i)
            u_par_i         = u_par_arr[mask]
            u_par_scaled_i  = u_par_i / np.mean(R_i)

            u_perp_hist.append(
                (np.quantile(u_perp_i, 0.84) - np.quantile(u_perp_i, 0.16)) / 2.
            )
            u_perp_scaled_hist.append(
                (np.quantile(u_perp_scaled_i, 0.84) - np.quantile(u_perp_scaled_i, 0.16)) / 2.
            )
            u_par_hist.append(
                (np.quantile(u_par_i, 0.84) - np.quantile(u_par_i, 0.16)) / 2.
            )
            u_par_scaled_hist.append(
                (np.quantile(u_par_scaled_i, 0.84) - np.quantile(u_par_scaled_i, 0.16)) / 2.
            )

        resolution_hists[key] = {
            'u_perp_resolution':        np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_perp_hist),
            'u_perp_scaled_resolution': np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_perp_scaled_hist),
            'u_par_resolution':         np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_par_hist),
            'u_par_scaled_resolution':  np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=u_par_scaled_hist),
            'R':                        np.histogram(qT_hist, bins=x_n, range=(0, max_x), weights=R_hist),
        }

    # ── Save outputs ──
    resolutions_path = osp.join(model_dir, 'best.resolutions')
    metrics_path     = osp.join(model_dir, 'metrics_eval.json')

    utils.save(resolution_hists, resolutions_path)
    utils.save_dict_to_json({'loss': mean_loss, 'epoch': epoch}, metrics_path)

    print(f"\nSaved:")
    print(f"  Resolution file : {resolutions_path}")
    print(f"  Metrics JSON    : {metrics_path}")
    print(f"\nYou can now point compare_response_resolution.ipynb at:")
    print(f"  {model_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a distilled student model and generate resolution files."
    )
    parser.add_argument('--ckpts', type=str, required=True,
                        help="Path to the student checkpoint directory (containing best.pth.tar)")
    parser.add_argument('--option', type=str, required=True,
                        choices=['depth4_spaced', 'depth4_endpoints', 'depth2_matched'],
                        help="Student architecture option (must match what was used for training)")
    parser.add_argument('--data', type=str, default='data_ttbar',
                        help="Data folder name relative to project root (default: data_ttbar)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for evaluation (default: 32)")

    args = parser.parse_args()
    run_evaluation(args)
