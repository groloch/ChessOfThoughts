import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import chess

from ..config import ChessEvaluationDataConfig


class ChessEvaluationDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        files = os.listdir(self.data_path)
        files = [filename for filename in files if filename.endswith('.parquet')]
        files = [os.path.join(self.data_path, filename) for filename in files]

        df_list = []
        for filepath in files:
            df = pd.read_parquet(filepath)
            df_list.append(df)
        self.df = pd.concat(df_list, ignore_index=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        fen = row['fen']
        line = row['line']
        score = row['score']
        cp = row['cp']
        mate = row['mate']
        turn = row['turn']

        move = chess.Move.from_uci(line)
        from_ = move.from_square
        to_ = move.to_square
        move_tensor = torch.zeros(64, 67)
        promotion_rank = {
            chess.QUEEN: 63,
            chess.ROOK: 64,
            chess.BISHOP: 65,
            chess.KNIGHT: 66
        }
        if not turn:
            from_ = 63 - from_
            to_ = 63 - to_
        if (ptype:=move.promotion) is not None:
            move_tensor[from_, promotion_rank[ptype]] = 1
        else:
            move_tensor[from_, to_] = 1

        board = chess.Board(fen)
        if not turn:
            board = board.mirror()
            score = -score

        position = torch.zeros(8, 8, 18, dtype=torch.float32)
        pieces_indices = {
            chess.KING: 0,
            chess.QUEEN: 1,
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4,
            chess.PAWN: 5
        }
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(i, j))
                if piece is not None:
                    c = pieces_indices[piece.piece_type]
                    if piece.color == chess.BLACK:
                        c += 6
                    position[i, j, c] = 1

        if (ep:=board.ep_square) is not None:
            position[chess.square_file(ep), chess.square_rank(ep), 12] = 1
        if board.castling_rights & chess.BB_H1:
            position[:, :, 13] = torch.ones(8, 8)
        if board.castling_rights & chess.BB_A1:
            position[:, :, 14] = torch.ones(8, 8)
        if board.castling_rights & chess.BB_H8:
            position[:, :, 15] = torch.ones(8, 8)
        if board.castling_rights & chess.BB_A8:
            position[:, :, 16] = torch.ones(8, 8)

        rule50count = board.halfmove_clock / 50.0
        position[:, :, 17] = torch.ones(8, 8) * rule50count

        turn = torch.tensor(turn, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)
        cp = torch.tensor(cp, dtype=torch.float32)
        mate = torch.tensor(mate, dtype=torch.float32)

        position = position.view(64, 18)

        return position, turn, move_tensor, score, cp, mate


def create_chess_evaluation_dataloaders(config: ChessEvaluationDataConfig):

    data_path = config.data_path
    seed = config.seed
    valid_ratio = config.valid_ratio
    batch_size = config.batch_size
    num_workers = config.num_workers

    dataset = ChessEvaluationDataset(data_path)

    indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    valid_number = int(len(dataset) * valid_ratio)

    train_dataset = Subset(dataset, indices[valid_number:])
    valid_dataset = Subset(dataset, indices[:valid_number])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, valid_loader, train_dataset, valid_dataset


if __name__ == '__main__':
    config = {
        'data_path': sys.argv[1],
        'seed': 12234,
        'valid_ratio': 0.1,
        'batch_size': 4096,
        'num_workers': 16
    }
    cfg = ChessEvaluationDataConfig(**config)
    tl, vl, ts, vs = create_chess_evaluation_dataloaders(cfg)

    print(len(ts), len(vs))

    for batch in tl:
        pos, turn, move, score, _, _ = batch
        print(pos.shape, turn.shape, move.shape, score.shape)
        print(pos.dtype, turn.dtype, move.dtype, score.dtype)
        print(move.sum())
        break