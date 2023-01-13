import functools
import logging
import os
import random
import sys

import numpy as np
import torch


def seed_torch(seed=1999):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name="log", logging_level="debug") -> logging.Logger:
    if logging_level == "debug":
        logging_level = logging.DEBUG
    elif logging_level == "info":
        logging_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
