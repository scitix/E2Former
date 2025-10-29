"""
E2Former Source Code

Organized structure:
- models/: Main model implementations (E2former, E2FormerBackbone)
- layers/: Neural network layers (attention, embeddings, blocks, interactions)
- core/: Base classes and utilities (base_modules, module_utils, e2former_utils)
- configs/: Configuration classes
- utils/: General utilities
- wigner6j/: Wigner 6j symbol and tensor product operations
"""

# Main model imports
from .models import E2former, E2FormerBackbone

# For backward compatibility
from .e2former import *

__all__ = [
    "E2former",
    "E2FormerBackbone",
]