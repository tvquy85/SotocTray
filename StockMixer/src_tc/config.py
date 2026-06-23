from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class ExperimentConfig:
    dataset_root: str
    market_name: str
    stock_num: int
    lookback_length: int
    fea_num: int
    steps: int
    valid_index: int
    test_index: int
    market_dim: int
    scale_factor: int
    epochs: int
    lr: float
    weight_decay: float
    alpha_rank: float
    beta_reg: float
    gamma_net: float
    rho_concentration: float
    tau: float
    train_cost_bps: float
    eval_cost_bps: float
    topk: int
    seed: int
    use_amp: bool
    torch_compile: bool
    method: str
    output_dir: str
    rank_sample_size: Optional[int] = None
    speed_mode: bool = True
    early_stop_patience: int = 15
    dynamic_netrank: bool = True
    selection_backtest: str = "softmax"
    selection_metric: str = "net_sharpe"
    ablation_mode: str = "dynamic_both"
    static_alpha: float = 0.05
    static_lambda: float = 0.05
    random_context_seed: int = 12345


def load_config(path: str) -> ExperimentConfig:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)
