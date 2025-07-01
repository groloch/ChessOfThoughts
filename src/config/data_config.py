from dataclasses import dataclass


@dataclass
class ChessEvaluationDataConfig:
    data_path: str
    seed: int
    batch_size: int
    valid_ratio: float
    num_workers: int
