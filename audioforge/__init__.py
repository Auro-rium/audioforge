"""AudioForge.

Benchmark-grade audio ML system for:
- FSD50K multi-label audio event classification
- DCASE-style machine anomaly detection
- distributed PyTorch/Hugging Face training
"""

from audioforge.config import AudioForgeConfig, load_config

__all__ = ["AudioForgeConfig", "load_config"]

__version__ = "0.1.0"
