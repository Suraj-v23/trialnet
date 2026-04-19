"""
TrialNet — A Novel AI Model That Learns From Its Mistakes

TrialNet combines traditional backpropagation with a new "Try and Learn" mechanism:
- Error Memory Bank: Remembers every mistake with full context
- Mistake Pattern Analyzer: Finds recurring failure patterns
- Perturbation Explorer: Experiments with weight changes to fix mistakes
- Prioritized Mistake Replay: Re-trains on hardest failures

Built entirely from scratch with NumPy. No TensorFlow. No PyTorch.
"""

__version__ = "0.1.0"
__author__ = "Suraj"

from trialnet.model import TrialNet

__all__ = ["TrialNet"]
