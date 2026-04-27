"""
scpFormer: A transformer-based foundation model for single-cell proteomics.
"""

import logging

__version__ = "0.1.0"

logger = logging.getLogger("scpformer")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .model import ScpFormerConfig, ScpFormerModel, ScpFormerForClassification
