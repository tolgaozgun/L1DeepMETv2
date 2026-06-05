"""
train_student_baseline.py

Trains a student model (depth4_spaced, depth4_endpoints, or depth2_matched)
from scratch on the TTbar dataset WITHOUT a teacher (no knowledge distillation).
This establishes the baseline performance of the smaller architectures to verify
how much distillation actually helps.

Run from the project root:
    python train_student_baseline.py --option depth4_spaced --data data_ttbar --ckpts distillation/outputs/baselines/...
"""

import argparse
import sys
import os
import os.path as osp
import numpy as np
import warnings
import json
from time import strftime, gmtime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_cluster import radius_graph
from tqdm import tqdm

warnings.simplefilter('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from model.student_net_depth4 import Net as StudentNetDepth4
from model.student_net_depth2 import Net as StudentNetDepth2
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import utils

# ── Constants ───────────────────────────────────────────────────────────────
N_FEATURES_CONT = 6
N_FEATURES_CAT  = 2
SCALE_MOMENTUM  = 128
EPOCHS          = 100
DELTA_R         = 0.4
DELTA_R_DZ      = 0.3 # not used but required by evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--option', required=True,
                    choices=['depth4_spaced', 'depth4_endpoints', 'depth2_matched'],
                    help="Student architecture to use")
parser.add_argument('--data', default='data_ttbar', help="Data directory")
parser.add_argument('--ckpts', required=True, help="Output checkpoints directory")
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.001, type=float)

# ── Wrapper to strip embeddings ─────────────────────────────────────────────
class StudentBaselineWrapper(nn.Module):
    """
    Student nets return (weights, embeddings) for distillation.
    For baseline training, we only want the weights to compute standard loss.
    """
    def __init__(self, student_model):
        super().__init__()
        self.model = student_model
        
    def forward(self, *args, **kwargs):
        weights, _ = self.model(*args, **kwargs)
        return weights


# ── Training logic ──────────────────────────────────────────────────────────
def train_epoch(model, device, optimizer, scheduler, loss_fn, dataloader):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader), desc='Training') as t:
        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)

            x_cont = data.x[:, :N_FEATURES_CONT]
            x_cat  = data.x[:, N_FEATURES_CONT:].long()
            etaphi = torch.cat([data.x[:, 3][:, None], data.x[:, 4][:, None]], dim=1)
            
            edge_index = radius_graph(etaphi, r=DELTA_R, batch=data.batch,
                                      loop=False, max_num_neighbors=255)
            
            result = model(x_cont, x_cat, edge_index, data.batch)
            
            loss = loss_fn(result, data.x, data.y, data.batch, SCALE_MOMENTUM)
            loss.backward()
            optimizer.step()
            
            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            
    scheduler.step(np.mean(loss_avg_arr))
    return np.mean(loss_avg_arr)


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    norm = torch.tensor([1. / SCALE_MOMENTUM] * 3 + [1., 1., 1.]).to(device)

    # ── Data ──
    print(f"Loading data from {args.data}...")
    dataloaders = data_loader.fetch_dataloader(
        data_dir=osp.join(project_root, args.data),
        batch_size=args.batch_size,
        validation_split=.2,
    )
    train_dl = dataloaders['train']
    test_dl  = dataloaders['test']
    print(f'Training dataloader: {len(train_dl)}, Test dataloader: {len(test_dl)}')

    # ── Model ──
    print(f"Setting up student model: {args.option}...")
    if args.option in ['depth4_spaced', 'depth4_endpoints']:
        base_model = StudentNetDepth4(continuous_dim=N_FEATURES_CONT, categorical_dim=N_FEATURES_CAT, norm=norm).to(device)
    elif args.option == 'depth2_matched':
        base_model = StudentNetDepth2(continuous_dim=N_FEATURES_CONT, categorical_dim=N_FEATURES_CAT, norm=norm).to(device)
    else:
        raise ValueError(f"Invalid option: {args.option}")

    model = StudentBaselineWrapper(base_model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-4, cycle_momentum=False)
    
    loss_fn = net.loss_fn_response_tune
    metrics = net.metrics

    # ── Output setup ──
    model_dir = osp.join(project_root, args.ckpts)
    os.makedirs(model_dir, exist_ok=True)
    loss_log = open(osp.join(model_dir, 'loss.log'), 'w')
    loss_log.write('# baseline loss log started ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
    loss_log.write('epoch,train_loss,val_loss\n')
    loss_log.flush()

    best_val_loss = 1e7

    # ── Epoch loop ──
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}] best_val={best_val_loss:.6f}")
        if '_last_lr' in scheduler.state_dict():
            print(f"  LR: {scheduler.state_dict()['_last_lr'][0]}")

        train_loss = train_epoch(model, device, optimizer, scheduler, loss_fn, train_dl)
        
        # Evaluate using evaluate.py logic
        test_metrics, resolutions, MET_arr = evaluate(
            model, device, loss_fn, test_dl, metrics, DELTA_R, DELTA_R_DZ, model_dir, epoch, save_METarr=True)
            
        val_loss = test_metrics['loss']
        print(f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        loss_log.write(f'{epoch},{train_loss:.8f},{val_loss:.8f}\n')
        loss_log.flush()

        is_best = (val_loss <= best_val_loss)
        if is_best:
            print('  → Found new best loss!')
            best_val_loss = val_loss
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'sched_dict': scheduler.state_dict()},
                                  is_best=True, checkpoint=model_dir)
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))

    loss_log.close()
    print("Baseline training complete!")

if __name__ == '__main__':
    run_training(parser.parse_args())
