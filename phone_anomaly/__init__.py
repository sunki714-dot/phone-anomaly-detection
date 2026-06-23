"""Phone screen anomaly detection (PatchCore + PaDiM) with grading & price estimation.

Label-free defect detection for used-phone screens: train only on normal (정상)
images, then flag anything that deviates — including defect types never seen
during training (open-set).
"""

__version__ = "0.1.0"

from .features import FeatureExtractor
from .models import PatchCore, PaDiM, zfuse

__all__ = ["FeatureExtractor", "PatchCore", "PaDiM", "zfuse", "__version__"]
