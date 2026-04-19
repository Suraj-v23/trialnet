"""
trial_learner.py — Try-and-Learn Orchestrator for TrialNet 🆕 NOVEL

The brain of the dual learning system. Orchestrates the full Try-and-Learn
pipeline by coordinating:
1. Mistake capture → Error Memory Bank
2. Pattern analysis → Mistake Pattern Analyzer
3. Weight exploration → Perturbation Explorer
4. Mistake replay → Targeted retraining on hard samples
5. Weight merging → Combining traditional and trial-based updates

This is what makes TrialNet fundamentally different from traditional models.
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from trialnet.core.layers import Layer
from trialnet.core.losses import Loss
from trialnet.core.tensor import Tensor
from trialnet.learning.error_memory import ErrorMemoryBank
from trialnet.learning.mistake_analyzer import MistakePatternAnalyzer, MistakeReport
from trialnet.learning.perturbation import PerturbationExplorer


class TrialLearner:
    """
    The Try-and-Learn Orchestrator.

    Lifecycle per training step:
    1. CAPTURE: collect mistakes from current predictions
    2. ANALYZE: discover patterns in mistakes
    3. EXPLORE: try perturbation-based weight corrections
    4. REPLAY: re-train on hardest mistakes
    5. MERGE: blend trial corrections with traditional gradients
    """

    def __init__(
        self,
        num_classes: int = 10,
        memory_capacity: int = 5000,
        similarity_threshold: float = 0.85,
        perturbation_scale: float = 0.01,
        n_perturbation_candidates: int = 5,
        replay_batch_size: int = 32,
        analyze_every: int = 5,
        explore_every: int = 3,
        trial_weight: float = 0.3,
    ):
        """
        Args:
            num_classes: Number of output classes
            memory_capacity: Max mistakes to store
            similarity_threshold: Threshold for deduplicating mistakes
            perturbation_scale: Initial scale for weight perturbation
            n_perturbation_candidates: Number of perturbation trials per step
            replay_batch_size: Number of mistakes to replay per step
            analyze_every: Run full analysis every N steps
            explore_every: Run perturbation exploration every N steps
            trial_weight: Weight of trial-based updates vs traditional (0-1)
        """
        # Core components
        self.memory_bank = ErrorMemoryBank(
            capacity=memory_capacity,
            similarity_threshold=similarity_threshold,
        )
        self.analyzer = MistakePatternAnalyzer(num_classes=num_classes)
        self.explorer = PerturbationExplorer(
            initial_scale=perturbation_scale,
            n_candidates=n_perturbation_candidates,
        )

        # Configuration
        self.num_classes = num_classes
        self.replay_batch_size = replay_batch_size
        self.analyze_every = analyze_every
        self.explore_every = explore_every
        self.trial_weight = trial_weight  # α coefficient for blending

        # State
        self._step_count = 0
        self._latest_report: Optional[MistakeReport] = None
        self._enabled = True

        # Metrics
        self.metrics_history: List[Dict] = []

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def step(
        self,
        layers: List[Layer],
        loss_fn: Loss,
        inputs: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        forward_fn: Callable,
        backward_fn: Callable,
    ) -> Dict:
        """
        Execute one Try-and-Learn step.

        This is called AFTER the traditional backprop step, providing
        a second learning signal.

        Args:
            layers: Network layers (for weight access)
            loss_fn: Loss function
            inputs: Current batch inputs
            predictions: Model's predictions for this batch
            targets: True labels
            forward_fn: Function to run forward pass (for evaluation)
            backward_fn: Function to run backward pass (for replay training)

        Returns:
            Dict with metrics from this step
        """
        if not self._enabled:
            return {'enabled': False}

        self._step_count += 1
        step_metrics = {'step': self._step_count}

        # ──────────────────────────────────────────────────────────────
        # PHASE 1: CAPTURE — Record mistakes from current batch
        # ──────────────────────────────────────────────────────────────
        per_sample_losses = loss_fn.get_per_sample_loss(predictions, targets)
        new_mistakes = self.memory_bank.record_batch_mistakes(
            inputs=inputs,
            predictions=predictions,
            targets=targets,
            per_sample_losses=per_sample_losses,
        )
        step_metrics['new_mistakes'] = len(new_mistakes)
        step_metrics['memory_size'] = self.memory_bank.size

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: ANALYZE — Find patterns in mistakes
        # ──────────────────────────────────────────────────────────────
        if self._step_count % self.analyze_every == 0:
            predicted_classes = np.argmax(predictions, axis=1)
            if targets.ndim == 1:
                target_classes = targets.astype(int)
            else:
                target_classes = np.argmax(targets, axis=1)
            accuracy = float(np.mean(predicted_classes == target_classes))

            self._latest_report = self.analyzer.analyze(
                self.memory_bank,
                recent_accuracy=accuracy,
            )
            step_metrics['analysis'] = {
                'severity': self._latest_report.overall_severity,
                'patterns_found': len(self._latest_report.patterns),
                'top_confusions': self._latest_report.top_confusions[:3],
                'hardest_classes': self._latest_report.hardest_classes,
            }

        # ──────────────────────────────────────────────────────────────
        # PHASE 3: EXPLORE — Try perturbation-based corrections
        # ──────────────────────────────────────────────────────────────
        if self._step_count % self.explore_every == 0 and self.memory_bank.size > 0:
            # Get trainable parameters
            params = []
            for layer in layers:
                params.extend(layer.get_params())

            # Create evaluation function that tests on mistake samples
            def eval_on_mistakes() -> float:
                batch = self.memory_bank.get_mistakes_as_batch(
                    n=min(self.replay_batch_size, self.memory_bank.size)
                )
                if batch is None:
                    return float('inf')
                mistake_inputs, mistake_targets = batch
                mistake_preds = forward_fn(mistake_inputs)
                return float(loss_fn(mistake_preds, mistake_targets))

            explore_result = self.explorer.explore(
                params=params,
                eval_fn=eval_on_mistakes,
                mistake_report=self._latest_report,
            )
            step_metrics['exploration'] = explore_result

        # ──────────────────────────────────────────────────────────────
        # PHASE 4: REPLAY — Re-train on hardest mistakes (periodically)
        # ──────────────────────────────────────────────────────────────
        replay_metrics = {'replayed': 0}
        if self._step_count % 5 == 0:
            replay_metrics = self._mistake_replay(
                layers=layers,
                loss_fn=loss_fn,
                forward_fn=forward_fn,
                backward_fn=backward_fn,
            )
        step_metrics['replay'] = replay_metrics

        # ──────────────────────────────────────────────────────────────
        # PHASE 5: MAINTENANCE — Decay, cleanup, check corrections
        # ──────────────────────────────────────────────────────────────
        if self._step_count % 20 == 0:
            self.memory_bank.decay_priorities(factor=0.98)
            self._check_corrections(forward_fn)

        if self._step_count % 50 == 0:
            self.memory_bank.cleanup_corrected(keep_ratio=0.05)
            self.memory_bank.update_history()

        # Store metrics
        self.metrics_history.append(step_metrics)

        return step_metrics

    def _mistake_replay(
        self,
        layers: List[Layer],
        loss_fn: Loss,
        forward_fn: Callable,
        backward_fn: Callable,
    ) -> Dict:
        """
        Replay training on mistakes from the memory bank.

        Uses prioritized sampling — the hardest mistakes get replayed
        more frequently.
        """
        if self.memory_bank.size == 0:
            return {'replayed': 0}

        # Sample mistakes with priority weighting
        replay_samples = self.memory_bank.sample_weighted(n=self.replay_batch_size)
        if not replay_samples:
            return {'replayed': 0}

        # Create batch from mistake samples
        replay_inputs = np.array([m.input_data for m in replay_samples])
        replay_targets = np.array([m.correct_class for m in replay_samples])

        # Forward pass on replay batch
        replay_preds = forward_fn(replay_inputs)

        # Compute loss
        replay_loss = float(loss_fn(replay_preds, replay_targets))

        # Backward pass with trial weight
        # The gradients are scaled by trial_weight to blend with traditional gradients
        grad_output = loss_fn.backward(replay_preds, replay_targets) * self.trial_weight
        backward_fn(grad_output)

        # Update correction attempts
        for m in replay_samples:
            m.correction_attempts += 1

        return {
            'replayed': len(replay_samples),
            'replay_loss': replay_loss,
            'trial_weight': self.trial_weight,
        }

    def _check_corrections(self, forward_fn: Callable):
        """
        Check if previously recorded mistakes are now corrected.
        Verifies by re-predicting on stored mistake samples.
        """
        if self.memory_bank.size == 0:
            return

        # Sample a subset to check
        check_samples = self.memory_bank.sample_random(n=min(100, self.memory_bank.size))
        check_inputs = np.array([m.input_data for m in check_samples])
        check_preds = forward_fn(check_inputs)
        pred_classes = np.argmax(check_preds, axis=1)

        corrected_ids = []
        for m, pred_class in zip(check_samples, pred_classes):
            if pred_class == m.correct_class:
                corrected_ids.append(m.id)

        if corrected_ids:
            self.memory_bank.mark_corrected(corrected_ids)

    def adapt_trial_weight(self, traditional_loss: float, trial_loss: float):
        """
        Adaptively adjust the weight between traditional and trial learning.

        If trial learning is helping (reducing loss), increase its weight.
        If it's hurting, decrease it.
        """
        if trial_loss < traditional_loss * 0.95:
            # Trial is helping — increase weight
            self.trial_weight = min(self.trial_weight * 1.05, 0.5)
        elif trial_loss > traditional_loss * 1.05:
            # Trial is hurting — decrease weight
            self.trial_weight = max(self.trial_weight * 0.95, 0.05)

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics about the Try-and-Learn system."""
        memory_stats = self.memory_bank.get_stats()
        explorer_stats = self.explorer.get_stats()
        trend = self.analyzer.get_trend()

        return {
            'step': self._step_count,
            'trial_weight': self.trial_weight,
            'memory': memory_stats,
            'exploration': explorer_stats,
            'trend': trend,
            'latest_report': {
                'severity': self._latest_report.overall_severity if self._latest_report else 0,
                'patterns': len(self._latest_report.patterns) if self._latest_report else 0,
                'hardest_classes': self._latest_report.hardest_classes if self._latest_report else [],
            } if self._latest_report else None,
        }

    def get_mistake_report(self) -> Optional[MistakeReport]:
        """Get the latest mistake analysis report."""
        return self._latest_report
