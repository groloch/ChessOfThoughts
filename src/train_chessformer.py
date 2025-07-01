import os
import shutil

import mlflow
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
import numpy as np

from .config import ChessEvaluationDataConfig, ChessFormerConfig, ChessformerPretrainingConfig
from .models import create_chessformer_model
from .data import create_chess_evaluation_dataloaders
from .utils import score_to_cp, create_logdir, trainable_parameters


def train_epoch(
        model: Module, 
        dataloader: DataLoader,
        accumulation_steps: int,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        epoch: int,
        logging_steps: int,
        logging_dir: str):

    model.train()
    model.bfloat16()
    model.to('cuda')
    model.compile(mode='max-autotune-no-cudagraphs')

    batch_size = dataloader.batch_size

    pbar = tqdm(total=1 + len(dataloader) // accumulation_steps)
    pbar.set_description(f'Epoch {epoch}: Score mse: 0.0; Move bce 0.0')

    window_size = 500
    score_loss_window = np.zeros(500)
    move_loss_window = np.zeros(500)
    cpl_window = np.zeros(500)
    acc_window = np.zeros(500)
    num_samples = 0

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for i, batch in enumerate(dataloader):
        pos, _, move, score, _, _ = batch
        
        pos = pos.to(torch.bfloat16).to('cuda')
        move = move.to(torch.bfloat16)
        score = score.to(torch.bfloat16).to('cuda')

        policy_logits, value_logit = model(pos)
        
        policy_logits = policy_logits.view(batch_size, -1)
        move = move.view(batch_size, -1)

        policy_move = torch.ones_like(policy_logits, device='cpu', dtype=torch.bfloat16) * -1
        policy_move[:, torch.argmax(policy_logits, dim=0)] = 1

        acc = (policy_move == move).sum() / batch_size
        acc_window[i % window_size] = acc.item()

        move = move.to('cuda')

        predicted_score = nn.functional.sigmoid(value_logit)
        predicted_score = predicted_score * 2 - 1

        # print(torch.max(value_logit), torch.min(value_logit), torch.mean(value_logit, dim=0).mean(), torch.max(predicted_score), torch.min(predicted_score), torch.mean(predicted_score, dim=0).mean())

        score_loss = mse_loss(predicted_score, score) / accumulation_steps
        move_loss = ce_loss(policy_logits, move) / accumulation_steps

        score_loss_window[i % window_size] = score_loss.item()
        move_loss_window[i % window_size] = move_loss.item()

        cp_target = score_to_cp(score)
        cp_pred = score_to_cp(predicted_score)
        cpl = torch.mean(torch.abs(cp_target - cp_pred)).item()
        cpl_window[i % window_size] = cpl

        num_samples += batch_size

        move_loss.backward(retain_graph=True)
        score_loss.backward()

        if i % accumulation_steps == 0 and i != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for p in model.parameters():
                p.grad = None

        if i % accumulation_steps == 0:
            mlflow.log_metric('score_mse_loss', score_loss.item() * accumulation_steps, step=epoch*len(dataloader)+i)
            mlflow.log_metric('move_ce_loss', move_loss.item() * accumulation_steps, step=epoch*len(dataloader)+i)

            mlflow.log_metric('score_mse_loss_avg', score_loss_window.sum()/min(i+1, window_size) * accumulation_steps, step=epoch * len(dataloader) + i)
            mlflow.log_metric('move_ce_loss_avg', move_loss_window.sum()/min(i+1, window_size) * accumulation_steps, step=epoch * len(dataloader) + i)

            mlflow.log_metric('cpl', cpl, step=epoch*len(dataloader)+i)
            mlflow.log_metric('cpl_avg', cpl_window.sum()/min(i+1, window_size), step=epoch*len(dataloader)+i)

            mlflow.log_metric('accuracy', acc.item(), step=epoch*len(dataloader)+i)
            mlflow.log_metric('accuracy_avg', acc_window.sum()/min(i+1, window_size), step=epoch*len(dataloader)+i)

            desc = f'Epoch {epoch}: '
            desc += f'Score mse: {score_loss_window.sum()/min(i+1, window_size)*accumulation_steps:.4e}; '
            desc += f'Move ce {move_loss_window.sum()/min(i+1, window_size)*accumulation_steps:.4e}; '
            desc += f'CPL {cpl_window.sum()/min(i+1, window_size):.4e} '
            desc += f'Accuracy {acc_window.sum()/min(i+1, window_size):.4e}'
            pbar.set_description(desc, refresh=False)
            pbar.update(1)

        if (i // accumulation_steps) % logging_steps == 0 and i // accumulation_steps != 0:
            torch.save(model.state_dict(), os.path.join(logging_dir, f'checkpoint_{(i // accumulation_steps)}.pt'))
            training_state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'step': i
            }
            torch.save(training_state, os.path.join(logging_dir, f'training_{(i // accumulation_steps)}.pt'))

def pretrain_chessformer(config, config_path):
    data_config = ChessEvaluationDataConfig(**config['data'])
    model_config = ChessFormerConfig(**config['model'])
    train_config = ChessformerPretrainingConfig(**config['training'])

    mlflow.set_tracking_uri(f'file://{os.path.abspath(train_config.mlflow_tracking_dir)}')
    mlflow.set_experiment('chessformer_pretraining')
    mlflow.start_run(run_name=train_config.name, description='')

    logging_dir = create_logdir(train_config.logging_dir, train_config.name)
    print(f"Logging files in: {logging_dir}")
    config_dest = os.path.join(logging_dir, os.path.basename(config_path))
    shutil.copyfile(config_path, config_dest)

    model = create_chessformer_model(model_config)
    train_loader, valid_loader, ts, _ = create_chess_evaluation_dataloaders(data_config)

    print(f'Model parameters: {trainable_parameters(model):,}')

    print(f'Training on {len(ts):,} positions')

    training_steps = 1 + train_config.epochs * len(train_loader) // train_config.accumulation_steps

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1/train_config.warmup_steps,
        end_factor=1.0,
        total_iters=train_config.warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1/train_config.lr_reduce_ratio,
        total_iters=training_steps-train_config.warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer)

    for e in range(train_config.epochs):
        train_epoch(
            model,
            train_loader,
            train_config.accumulation_steps,
            optimizer,
            scheduler,
            e,
            train_config.logging_steps,
            logging_dir
        )

    torch.save(model.state_dict(), os.path.join(logging_dir, 'checkpoint_last.pt'))

    mlflow.end_run()
