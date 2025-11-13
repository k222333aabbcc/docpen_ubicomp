from enum import Enum

import torch

DEVICE = [0, 1, 2, 3]


class WeightsInitializer(str, Enum):
    Zeros = "zeros"
    He = "he"
    Xavier = "xavier"
