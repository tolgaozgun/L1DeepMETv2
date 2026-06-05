"""
Knowledge Distillation training script for L1DeepMETv2.
Trains a student model (depth4_spaced, depth4_endpoints, or depth2_matched)
against a pre-trained teacher model using output, hint, and LSP distillation losses.

Run from the project root:
    python train_distillation.py --teacher_ckpt teacher-model/ckpts_dytt/best.pth.tar --option depth4_spaced --data data_dytt --ckpts distillation/outputs/...
"""

import argparse
import importlib
import sys
import os
import os.path as osp
import numpy as np
import warnings
from time import strftime, gmtime

import torch
import torch.optim as optim
from torch_cluster import radius_graph
from tqdm import tqdm

warnings.simplefilter('ignore')

# ── Path setup ──────────────────────────────────────────────────────────────
# This script lives at the project root, so 'model', 'utils', and
# 'data_loader' all import naturally. The only trick needed is loading
# the teacher from its own separate repo without polluting the student's
# 'model' namespace.

project_root = os.path.dirname(os.path.abspath(__file__))

# 1. Insert teacher-model repo first so 'model.net' resolves to the teacher.
teacher_dir = osp.join(project_root, 'teacher-model')
sys.path.insert(0, teacher_dir)
from model.net import Net as TeacherNet

# 2. Remove teacher-model from path and purge its cached 'model' modules
#    so subsequent imports resolve from the student repo.
sys.path.pop(0)
for key in list(sys.modules.keys()):
    if key == 'model' or key.startswith('model.'):
        del sys.modules[key]
importlib.invalidate_caches()

# 3. Student repo imports (project root is already on sys.path when run directly)
from model.student_net_depth4 import Net as StudentNetDepth4
from model.student_net_depth2 import Net as StudentNetDepth2
import model.data_loader as data_loader
import utils

from distillation.distillation_loss import total_distillation_loss

# ── Constants ───────────────────────────────────────────────────────────────
# Student / data pipeline (matching this repo's train.py)
N_FEATURES_CONT = 6
N_FEATURES_CAT  = 2

# Teacher (trained with 8 continuous + 3 categorical features in teacher-model/train.py)
TEACHER_N_FEATURES_CONT = 8
TEACHER_N_FEATURES_CAT  = 3

SCALE_MOMENTUM = 128
DELTA_R = 0.4
EPOCHS = 100


# ── Training step ────────────────────────────────────────────────────────────
def train_epoch(student, teacher, device, optimizer, scheduler, dataloader, option):
    """Run one full training epoch of distillation."""
    student.train()
    teacher.eval()

    loss_avg_arr = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont = data.x[:, :N_FEATURES_CONT]
            x_cat  = data.x[:, N_FEATURES_CONT:].long()
            etaphi = torch.cat([data.x[:, 3][:, None], data.x[:, 4][:, None]], dim=1)
            edge_index = radius_graph(etaphi, r=DELTA_R, batch=data.batch,
                                      loop=False, max_num_neighbors=255)

            # Teacher forward (no gradients).
            # The teacher expects TEACHER_N_FEATURES_CONT continuous + TEACHER_N_FEATURES_CAT categorical features.
            # Our dataset has fewer; pad the remainder with zeros.
            with torch.no_grad():
                if x_cont.shape[1] < TEACHER_N_FEATURES_CONT:
                    pad_c = torch.zeros(x_cont.shape[0],
                                        TEACHER_N_FEATURES_CONT - x_cont.shape[1],
                                        device=device)
                    x_cont_teacher = torch.cat([x_cont, pad_c], dim=1)
                else:
                    x_cont_teacher = x_cont
                if x_cat.shape[1] < TEACHER_N_FEATURES_CAT:
                    pad_k = torch.zeros(x_cat.shape[0],
                                        TEACHER_N_FEATURES_CAT - x_cat.shape[1],
                                        dtype=torch.long, device=device)
                    x_cat_teacher = torch.cat([x_cat, pad_k], dim=1)
                else:
                    x_cat_teacher = x_cat
                t_out, t_embs = teacher(x_cont_teacher, x_cat_teacher, edge_index, data.batch)

            # Student forward
            s_out, s_embs = student(x_cont, x_cat, edge_index, data.batch)

            loss, loss_dict = total_distillation_loss(
                student_outputs=s_out,
                teacher_outputs=t_out,
                student_embs=s_embs,
                teacher_embs=t_embs,
                batch=data.batch,
                alpha=1.0,   # output logit weight
                beta=1.0,    # feature hint weight
                gamma=0.5,   # LSP weight
                option=option,
            )

            loss.backward()
            optimizer.step()

            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    if not loss_avg_arr:
        raise RuntimeError("Training dataloader is empty — no batches were processed. "
                           "Check that the data directory exists and contains processed .pt files.")

    scheduler.step(np.mean(loss_avg_arr))
    return np.mean(loss_avg_arr), loss_dict


# ── Validation step ──────────────────────────────────────────────────────────
def validate_epoch(student, teacher, device, dataloader, option):
    """Compute distillation loss on the validation set (no backprop)."""
    student.eval()
    teacher.eval()

    loss_avg_arr = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            x_cont = data.x[:, :N_FEATURES_CONT]
            x_cat  = data.x[:, N_FEATURES_CONT:].long()
            etaphi = torch.cat([data.x[:, 3][:, None], data.x[:, 4][:, None]], dim=1)
            edge_index = radius_graph(etaphi, r=DELTA_R, batch=data.batch,
                                      loop=False, max_num_neighbors=255)

            with torch.no_grad():
                x_cont_t = (torch.cat([x_cont, torch.zeros(x_cont.shape[0],
                                        TEACHER_N_FEATURES_CONT - x_cont.shape[1], device=device)], dim=1)
                            if x_cont.shape[1] < TEACHER_N_FEATURES_CONT else x_cont)
                x_cat_t  = (torch.cat([x_cat, torch.zeros(x_cat.shape[0],
                                        TEACHER_N_FEATURES_CAT - x_cat.shape[1],
                                        dtype=torch.long, device=device)], dim=1)
                            if x_cat.shape[1] < TEACHER_N_FEATURES_CAT else x_cat)
                t_out, t_embs = teacher(x_cont_t, x_cat_t, edge_index, data.batch)
            s_out, s_embs = student(x_cont, x_cat, edge_index, data.batch)

            loss, _ = total_distillation_loss(
                student_outputs=s_out,
                teacher_outputs=t_out,
                student_embs=s_embs,
                teacher_embs=t_embs,
                batch=data.batch,
                alpha=1.0, beta=1.0, gamma=0.5,
                option=option,
            )
            loss_avg_arr.append(loss.item())

    return np.mean(loss_avg_arr)


# ── Main training loop ────────────────────────────────────────────────────────
def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # norm for input data (matching train.py scale_momentum=128)
    norm = torch.tensor([1. / SCALE_MOMENTUM] * 3 + [1., 1., 1.]).to(device)

    # ── Data ──
    print(f"Loading data from {args.data}...")
    dataloaders = data_loader.fetch_dataloader(
        data_dir=osp.join(project_root, args.data),
        batch_size=int(args.batch_size),
        validation_split=.2,
    )
    train_dl = dataloaders['train']
    test_dl  = dataloaders['test']
    print(f'Training dataloader: {len(train_dl)}, Test dataloader: {len(test_dl)}')

    if len(train_dl) == 0:
        raise RuntimeError(f"Training dataloader is empty. "
                           f"Check that '{args.data}/processed/' exists and contains .pt files.")
    if len(test_dl) == 0:
        raise RuntimeError(f"Test dataloader is empty. "
                           f"Check that '{args.data}/processed/' has enough data for validation split.")

    # ── Teacher ──
    teacher_ckpt = args.teacher_ckpt if osp.isabs(args.teacher_ckpt) else osp.join(project_root, args.teacher_ckpt)
    print(f"Loading Teacher from {teacher_ckpt}...")
    teacher = TeacherNet(continuous_dim=TEACHER_N_FEATURES_CONT, categorical_dim=TEACHER_N_FEATURES_CAT).to(device)
    checkpoint = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Student ──
    print(f"Setting up student with option: {args.option}...")
    if args.option in ['depth4_spaced', 'depth4_endpoints']:
        student = StudentNetDepth4(continuous_dim=N_FEATURES_CONT, categorical_dim=N_FEATURES_CAT, norm=norm).to(device)
    elif args.option == 'depth2_matched':
        student = StudentNetDepth2(continuous_dim=N_FEATURES_CONT, categorical_dim=N_FEATURES_CAT, norm=norm).to(device)
    else:
        raise ValueError(f"Invalid option: {args.option}")

    optimizer = optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-4, cycle_momentum=False)

    # ── Output directory and loss log ──
    model_dir = osp.join(project_root, args.ckpts)
    os.makedirs(model_dir, exist_ok=True)
    loss_log = open(osp.join(model_dir, 'loss.log'), 'w')
    loss_log.write('# distillation loss log started ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
    loss_log.write('epoch,train_loss,val_loss\n')
    loss_log.flush()

    best_val_loss = 1e7

    # ── Epoch loop ──
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}] best_val={best_val_loss:.6f}")
        if '_last_lr' in scheduler.state_dict():
            print(f"  LR: {scheduler.state_dict()['_last_lr'][0]}")

        train_loss, _ = train_epoch(student, teacher, device, optimizer, scheduler,
                                    train_dl, args.option)
        val_loss = validate_epoch(student, teacher, device, test_dl, args.option)

        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        loss_log.write(f'{epoch},{train_loss:.8f},{val_loss:.8f}\n')
        loss_log.flush()

        is_best = val_loss <= best_val_loss

        # Save last checkpoint
        utils.save_checkpoint(
            {'epoch': epoch, 'state_dict': student.state_dict(),
             'optim_dict': optimizer.state_dict(), 'sched_dict': scheduler.state_dict()},
            is_best=False, checkpoint=model_dir,
        )

        if is_best:
            best_val_loss = val_loss
            print("  ✓ New best model saved.")
            utils.save_checkpoint(
                {'epoch': epoch, 'state_dict': student.state_dict(),
                 'optim_dict': optimizer.state_dict(), 'sched_dict': scheduler.state_dict()},
                is_best=True, checkpoint=model_dir,
            )
            utils.save_dict_to_json({'loss': val_loss, 'epoch': epoch},
                                    osp.join(model_dir, 'metrics_val_best.json'))

        utils.save_dict_to_json({'loss': val_loss, 'epoch': epoch},
                                osp.join(model_dir, 'metrics_val_last.json'))

    loss_log.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Outputs saved to: {model_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training for L1DeepMETv2")
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help="Path to the pre-trained teacher checkpoint (.pth)")
    parser.add_argument('--option', type=str, required=True,
                        choices=['depth4_spaced', 'depth4_endpoints', 'depth2_matched'],
                        help="Student architecture / hint mapping strategy")
    parser.add_argument('--data', type=str, required=True,
                        help="Data folder name relative to project root (e.g. data_dytt)")
    parser.add_argument('--ckpts', type=str, default='ckpts',
                        help="Output checkpoint folder relative to project root")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size")
    parser.add_argument('--lr', default=0.1, type=float, help="Learning rate")
    parser.add_argument('--weight_decay', default=0.001, type=float, help="Weight decay")

    args = parser.parse_args()
    run_training(args)
