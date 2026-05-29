from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    Why this is needed:
    - Makes experiments more reproducible.
    - Helps compare scratch CNN vs BEATs/AST fairly.
    - Reduces random metric swings between runs.
    - Makes bug reproduction possible.

    Brutal caveat:
    Perfect reproducibility is not guaranteed on GPU because some CUDA kernels
    and distributed operations can still be nondeterministic. This gets us close
    enough for serious experiment tracking, not divine mathematical certainty.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not support deterministic mode depending on environment.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True