# evaluation/reproducibility.py
import os
import random
import numpy as np


def seed_everything(seed: int = 777, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
