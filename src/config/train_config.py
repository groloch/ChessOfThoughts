from dataclasses import dataclass


@dataclass
class ChessformerPretrainingConfig:
    epochs: int
    accumulation_steps: int
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_reduce_ratio: float
    mlflow_tracking_dir: str
    name: str
    logging_steps: int
    logging_dir: str
