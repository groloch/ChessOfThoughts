import os
import sys
import pandas as pd
import chess
import numpy as np


def filter_data(file, datadir, outputdir):
    print('Filtering', file)

    df = pd.read_parquet(f'{datadir}/{file}.parquet')
    df = df.drop(columns=['depth', 'knodes'])

    df['turn'] = df['fen'].apply(lambda fen: 'w' in fen)

    def cp_to_score(cp):
        return np.atan(cp / 111.714640912) / 1.5620688421

    def mate_to_score(mate):
        return np.where(mate < 0, -1.0, 1.0)

    cp_df = df.dropna(subset=['cp']).copy()
    mate_df = df.dropna(subset=['mate']).copy()

    cp_df['score'] = cp_df['cp'].apply(cp_to_score)
    mate_df['score'] = mate_df['mate'].apply(mate_to_score)

    df = pd.concat([cp_df, mate_df], ignore_index=True)

    def get_indices(df):
        turn_first = df.groupby('fen')['turn'].first()
        idxmax = df.groupby('fen')['score'].idxmax()
        idxmin = df.groupby('fen')['score'].idxmin()
        idx = idxmax.where(turn_first, idxmin)
        return idx.values

    idx = get_indices(df)
    df = df.loc[idx]

    def is_move_legal(fen, move_uci):
        try:
            position = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            return move in position.legal_moves
        except:
            return False

    df['line'] = df['line'].apply(lambda l: l.split(' ')[0])
    df['is_legal'] = df.apply(lambda row: is_move_legal(row['fen'], row['line']), axis=1)

    df = df[df['is_legal']]

    df = df.drop(columns=['is_legal'])

    df.to_parquet(f'{outputdir}/{file}_f.parquet')

def filter_dataset(datadir, outputdir):
    files = os.listdir(datadir)
    files = [f for f in files if f.endswith('.parquet')]

    for file in files:
        filter_data(file, datadir, outputdir)


if __name__ == '__main__':
    filter_dataset(sys.argv[1], sys.argv[2])