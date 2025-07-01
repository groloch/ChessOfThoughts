import os

import math
import torch
from torch import nn
import chess


def score_to_cp(score):
    if isinstance(score, torch.Tensor):
        cp = 111.714640912 * torch.tan(1.5620688421 * score)
        cp = torch.clip(cp, -1000, 1000)
    else:
        cp = 111.714640912 * math.tan(1.5620688421 * score)
    return cp

def mate_to_score(mate):
    if isinstance(mate, torch.Tensor):
        score = torch.where(mate < 0, -1.0, 1.0)
    else:
        score = -1.0 if mate < 0 else 1.0
    return score

def cp_to_score(cp):
    if isinstance(cp, torch.Tensor):
        score = torch.atan(cp / 111.714640912) / 1.5620688421
    else:
        score = math.atan(cp / 111.714640912) / 1.5620688421
    return score

def firstmove(line):
    return line.split(' ')[0]

def square_to_coords(square):
    x = chess.square_file(square)
    y = chess.square_rank(square)
    return x, y


def create_logdir(log_dir, subdir_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    new_subdir_name = subdir_name
    counter = 1
    while os.path.exists(os.path.join(log_dir, new_subdir_name)):
        new_subdir_name = f"{subdir_name}_{counter}"
        counter += 1

    subdir_path = os.path.join(log_dir, new_subdir_name)
    os.makedirs(subdir_path)

    return subdir_path

def trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)