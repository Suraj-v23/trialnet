"""
perturbation.py — Perturbation Explorer for TrialNet 🆕 NOVEL

The "Try" part of "Try and Learn". Instead of ONLY following gradients
(which can get stuck in local minima), the Perturbation Explorer
EXPERIMENTS with weight changes:

1. Random Perturbation — Add noise to weights, keep if it helps
2. Targeted Perturbation — Focus on weights connected to mistake patterns
3. Evolutionary Selection — Generate multiple variants, keep the best

This is inspired by:
- Evolutionary strategies (OpenAI ES)
- Random search optimization
- Simulated annealing
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Callable
from trialnet.core.tensor import Tensor
from trialnet.learning.mistake_analyzer import MistakeReport


class PerturbationExplorer:
    """
    Experiments with weight perturbations to find improvements
    that gradient descent alone might miss.

    The key insight: sometimes the gradient points in the wrong direction
    for specific hard examples, but a small random change could fix them.
    """

    def __init__(
        self,
        initial_scale: float = 0.01,
        min_scale: float = 0.001,
        max_scale: float = 0.1,
        n_candidates: int = 5,
        decay_rate: float = 0.995,
    ):
        """
        Args:
            initial_scale: Starting perturbation scale (std dev of noise)
            min_scale: Minimum perturbation scale
            max_scale: Maximum perturbation scale
            n_candidates: Number of perturbation candidates to try per step
            decay_rate: How quickly the perturbation scale decays
        """
        self.scale = initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_candidates = n_candidates
        self.decay_rate = decay_rate

        # Statistics
        self.total_trials = 0
        self.successful_trials = 0
        self.improvement_history: List[float] = []
        self.scale_history: List[float] = []

    @property
    def success_rate(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.successful_trials / self.total_trials

    def explore(
        self,
        params: List[Tensor],
        eval_fn: Callable[[], float],
        mistake_report: Optional[MistakeReport] = None,
    ) -> Dict:
        """
        Run perturbation exploration on model parameters.

        Args:
            params: List of model parameter tensors
            eval_fn: Function that returns current loss on mistake samples
            mistake_report: Optional analysis to guide targeted perturbation

        Returns:
            Dict with exploration results
        """
        # Get baseline loss
        baseline_loss = eval_fn()

        # Adjust scale based on mistake report
        if mistake_report:
            self.scale = np.clip(
                mistake_report.perturbation_strength,
                self.min_scale,
                self.max_scale
            )

        best_improvement = 0.0
        best_perturbations = None

        for trial in range(self.n_candidates):
            self.total_trials += 1

            # Save current weights
            for p in params:
                p.save_snapshot()

            # Apply perturbation
            if mistake_report and mistake_report.mistake_feature_importance is not None:
                self._apply_targeted_perturbation(params, mistake_report)
            else:
                self._apply_random_perturbation(params)

            # Evaluate
            perturbed_loss = eval_fn()
            improvement = baseline_loss - perturbed_loss

            if improvement > best_improvement:
                best_improvement = improvement
                # Save these perturbations (don't restore yet)
                best_perturbations = [p.data.copy() for p in params]
                self.successful_trials += 1

            # Restore weights for next trial
            for p in params:
                p.restore_snapshot()

        # Apply the best perturbation found (if any improvement)
        result = {
            'improved': best_improvement > 0,
            'improvement': float(best_improvement),
            'baseline_loss': float(baseline_loss),
            'best_loss': float(baseline_loss - best_improvement),
            'trials': self.n_candidates,
            'scale': self.scale,
        }

        if best_perturbations is not None and best_improvement > 0:
            for p, new_data in zip(params, best_perturbations):
                p.data = new_data
            self.improvement_history.append(best_improvement)
        else:
            self.improvement_history.append(0.0)

        # Decay scale
        self.scale = max(self.scale * self.decay_rate, self.min_scale)
        self.scale_history.append(self.scale)

        return result

    def _apply_random_perturbation(self, params: List[Tensor]):
        """Apply random Gaussian noise to all parameters."""
        for p in params:
            p.perturb(scale=self.scale)

    def _apply_targeted_perturbation(
        self,
        params: List[Tensor],
        mistake_report: MistakeReport
    ):
        """
        Apply perturbation focused on weights most related to mistake patterns.

        Strategy: Perturb more aggressively the weights connected to
        features/classes that appear in mistake patterns.
        """
        feature_importance = mistake_report.mistake_feature_importance
        focus_classes = mistake_report.focus_on_classes

        for i, p in enumerate(params):
            if p.data.ndim == 2:  # Weight matrix
                # Create perturbation mask based on feature importance
                mask = np.ones_like(p.data)

                # For the first layer: use feature importance to focus perturbation
                if i == 0 and feature_importance is not None:
                    n_features = min(feature_importance.shape[0], p.data.shape[0])
                    # Boost perturbation for important features (those causing mistakes)
                    feature_mask = np.ones(p.data.shape[0])
                    feature_mask[:n_features] = 1.0 + feature_importance[:n_features] * 3.0
                    mask = mask * feature_mask.reshape(-1, 1)

                # For the last layer: focus on confused classes
                if focus_classes and p.data.shape[1] >= max(focus_classes, default=0):
                    for cls in focus_classes:
                        if cls < p.data.shape[1]:
                            mask[:, cls] *= 2.0  # Perturb more for problematic classes

                p.perturb(scale=self.scale, mask=mask)
            else:
                # Bias or 1D params: uniform perturbation
                p.perturb(scale=self.scale)

    def evolutionary_step(
        self,
        params: List[Tensor],
        eval_fn: Callable[[], float],
        population_size: int = 10,
    ) -> Dict:
        """
        Evolutionary strategy step — generates a population of weight variants
        and keeps the best one.

        More expensive but can find better solutions for stubborn mistakes.
        """
        # Save original weights
        original_weights = [p.data.copy() for p in params]
        baseline_loss = eval_fn()

        candidates = []
        for _ in range(population_size):
            # Create a noisy variant
            noises = []
            for p in params:
                noise = np.random.randn(*p.data.shape) * self.scale
                p.data = original_weights[params.index(p)] + noise
                noises.append(noise)

            loss = eval_fn()
            candidates.append((loss, noises))

            # Restore
            for p, orig in zip(params, original_weights):
                p.data = orig.copy()

        # Find best candidate
        candidates.sort(key=lambda x: x[0])
        best_loss, best_noises = candidates[0]

        result = {
            'improved': best_loss < baseline_loss,
            'improvement': float(baseline_loss - best_loss),
            'baseline_loss': float(baseline_loss),
            'best_loss': float(best_loss),
            'population_size': population_size,
        }

        if best_loss < baseline_loss:
            # Apply the winning perturbation
            for p, noise, orig in zip(params, best_noises, original_weights):
                p.data = orig + noise
            self.successful_trials += 1

        self.total_trials += population_size

        return result

    def adaptive_scale_update(self):
        """
        Adaptively adjust perturbation scale based on success rate.

        If many trials succeed → scale is too small, increase it.
        If few trials succeed → scale is too large, decrease it.
        """
        if self.total_trials < 10:
            return

        recent_rate = self.success_rate

        if recent_rate > 0.5:
            # Too easy, increase scale to explore more
            self.scale = min(self.scale * 1.1, self.max_scale)
        elif recent_rate < 0.1:
            # Too hard, decrease scale for finer changes
            self.scale = max(self.scale * 0.9, self.min_scale)

    def get_stats(self) -> Dict:
        """Get exploration statistics."""
        return {
            'total_trials': self.total_trials,
            'successful_trials': self.successful_trials,
            'success_rate': self.success_rate,
            'current_scale': self.scale,
            'avg_improvement': float(np.mean(self.improvement_history)) if self.improvement_history else 0.0,
            'total_improvement': float(np.sum(self.improvement_history)),
        }
