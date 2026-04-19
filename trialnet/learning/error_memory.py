"""
error_memory.py — Error Memory Bank for TrialNet 🆕 NOVEL

The core innovation of TrialNet. Instead of forgetting mistakes after
a gradient update (like traditional models), this module explicitly
stores, indexes, and manages a memory of past mistakes.

Key features:
- Prioritized storage (high-confidence mistakes get higher priority)
- Similarity detection (groups similar mistakes)
- Decay mechanism (old corrected mistakes fade)
- Capacity management (fixed-size buffer with intelligent eviction)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class MistakeRecord:
    """
    A single mistake captured by the Error Memory Bank.

    This is NOT just a loss value — it's a complete snapshot of what went wrong,
    enabling the model to learn from the specific failure.
    """

    # The input that caused the mistake
    input_data: np.ndarray

    # What the model incorrectly predicted
    predicted_output: np.ndarray

    # What the correct output should have been
    correct_output: np.ndarray

    # How confident the model was (higher = more dangerous mistake)
    confidence: float

    # The loss value for this specific sample
    loss_value: float

    # What the model predicted vs what was correct (class indices)
    predicted_class: int
    correct_class: int

    # Internal layer activations when the mistake happened
    layer_activations: Optional[List[np.ndarray]] = None

    # When this mistake was recorded
    timestamp: float = field(default_factory=time.time)

    # Category of mistake (assigned by Mistake Pattern Analyzer)
    mistake_type: str = "unclassified"

    # How many times we've tried to correct this specific mistake
    correction_attempts: int = 0

    # Whether this mistake has been fixed (model now gets it right)
    is_corrected: bool = False

    # Priority score (higher = more important to fix)
    priority: float = 0.0

    # Unique identifier
    id: int = 0

    def compute_priority(self) -> float:
        """
        Compute priority score for this mistake.

        Priority is based on:
        1. Confidence (high confidence + wrong = very dangerous)
        2. Loss value (higher loss = more severe mistake)
        3. Correction attempts (many failed attempts = stubborn mistake)
        4. Recency (newer mistakes slightly prioritized)
        """
        confidence_factor = self.confidence * 2.0  # High-confidence errors are worst
        loss_factor = min(self.loss_value, 10.0)  # Cap to prevent outliers
        stubbornness = min(self.correction_attempts * 0.5, 5.0)  # Stubborn mistakes matter
        recency = max(0, 1.0 - (time.time() - self.timestamp) / 3600)  # Decay over 1 hour

        self.priority = confidence_factor + loss_factor + stubbornness + recency * 0.3
        return self.priority


class ErrorMemoryBank:
    """
    The Error Memory Bank — stores and manages a prioritized collection
    of mistakes made by the model.

    Unlike traditional training where errors are immediately forgotten after
    a weight update, the Error Memory Bank maintains a persistent memory
    of failures, enabling:

    1. Targeted retraining on hard examples
    2. Pattern detection across mistakes
    3. Perturbation-based exploration on specific failures
    4. Tracking whether mistakes are actually being corrected over time
    """

    def __init__(self, capacity: int = 10000, similarity_threshold: float = 0.85):
        """
        Args:
            capacity: Maximum number of mistakes to store
            similarity_threshold: Cosine similarity threshold for grouping similar mistakes
        """
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold

        # Main mistake storage
        self._memories: List[MistakeRecord] = []

        # Index by mistake type: correct_class → predicted_class
        self._confusion_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        # Statistics
        self._total_mistakes_seen = 0
        self._total_corrections = 0
        self._id_counter = 0

        # History for tracking progress
        self.mistake_count_history: List[int] = []
        self.correction_rate_history: List[float] = []

    @property
    def size(self) -> int:
        return len(self._memories)

    @property
    def is_full(self) -> bool:
        return len(self._memories) >= self.capacity

    def record_mistake(
        self,
        input_data: np.ndarray,
        predicted_output: np.ndarray,
        correct_output: np.ndarray,
        loss_value: float,
        layer_activations: Optional[List[np.ndarray]] = None
    ) -> MistakeRecord:
        """
        Record a single mistake into the memory bank.

        This is called during training whenever the model makes an incorrect prediction.
        """
        self._total_mistakes_seen += 1
        self._id_counter += 1

        # Determine predicted and correct classes
        predicted_class = int(np.argmax(predicted_output))
        correct_class = int(correct_output) if correct_output.ndim == 0 else int(np.argmax(correct_output) if correct_output.ndim > 0 and correct_output.shape[0] > 1 else correct_output[0])

        # Compute confidence (max probability of prediction)
        probs = self._softmax(predicted_output) if np.any(predicted_output > 1) else predicted_output
        confidence = float(np.max(probs))

        # Create mistake record
        mistake = MistakeRecord(
            input_data=input_data.copy(),
            predicted_output=predicted_output.copy(),
            correct_output=correct_output.copy() if isinstance(correct_output, np.ndarray) else np.array([correct_output]),
            confidence=confidence,
            loss_value=loss_value,
            predicted_class=predicted_class,
            correct_class=correct_class,
            layer_activations=[a.copy() for a in layer_activations] if layer_activations else None,
            id=self._id_counter,
        )
        mistake.compute_priority()

        # Check for duplicates
        if not self._is_duplicate(mistake):
            self._add_to_memory(mistake)

        return mistake

    def record_batch_mistakes(
        self,
        inputs: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        per_sample_losses: np.ndarray,
        layer_activations: Optional[List[np.ndarray]] = None
    ) -> List[MistakeRecord]:
        """
        Record mistakes from an entire batch.
        Only records samples where the prediction was WRONG.
        """
        predicted_classes = np.argmax(predictions, axis=1)

        # Handle targets
        if targets.ndim == 1:
            target_classes = targets.astype(int)
        else:
            target_classes = np.argmax(targets, axis=1)

        # Find mistakes (where prediction != target)
        mistake_mask = predicted_classes != target_classes
        mistake_indices = np.where(mistake_mask)[0]

        recorded = []
        for idx in mistake_indices:
            target_val = targets[idx]
            mistake = self.record_mistake(
                input_data=inputs[idx],
                predicted_output=predictions[idx],
                correct_output=target_val,
                loss_value=float(per_sample_losses[idx]),
                layer_activations=None,
            )
            recorded.append(mistake)

        return recorded

    def _add_to_memory(self, mistake: MistakeRecord):
        """Add a mistake to memory, evicting lowest-priority if full."""
        if self.is_full:
            self._evict_lowest_priority()

        self._memories.append(mistake)

        # Update confusion index
        key = (mistake.correct_class, mistake.predicted_class)
        self._confusion_index[key].append(len(self._memories) - 1)

    def _evict_lowest_priority(self):
        """Remove the lowest-priority mistake to make room."""
        if not self._memories:
            return

        # Find lowest priority
        min_idx = min(range(len(self._memories)), key=lambda i: self._memories[i].priority)

        # Remove from confusion index
        removed = self._memories[min_idx]
        key = (removed.correct_class, removed.predicted_class)
        if key in self._confusion_index:
            self._confusion_index[key] = [
                i for i in self._confusion_index[key] if i != min_idx
            ]

        self._memories.pop(min_idx)

        # Rebuild confusion index (indices shifted)
        self._rebuild_confusion_index()

    def _rebuild_confusion_index(self):
        """Rebuild the confusion index after eviction."""
        self._confusion_index.clear()
        for i, m in enumerate(self._memories):
            key = (m.correct_class, m.predicted_class)
            self._confusion_index[key].append(i)

    def _is_duplicate(self, new_mistake: MistakeRecord) -> bool:
        """
        Check if a very similar mistake already exists.
        Uses cosine similarity on input features.
        """
        if not self._memories:
            return False

        new_flat = new_mistake.input_data.flatten()
        new_norm = np.linalg.norm(new_flat)
        if new_norm == 0:
            return False

        # Only check same confusion pair
        key = (new_mistake.correct_class, new_mistake.predicted_class)
        indices = self._confusion_index.get(key, [])

        for idx in indices[-20:]:  # Check only recent similar ones for efficiency
            if idx < len(self._memories):
                existing_flat = self._memories[idx].input_data.flatten()
                existing_norm = np.linalg.norm(existing_flat)
                if existing_norm == 0:
                    continue
                similarity = np.dot(new_flat, existing_flat) / (new_norm * existing_norm)
                if similarity > self.similarity_threshold:
                    # Update priority of existing instead
                    self._memories[idx].correction_attempts += 1
                    self._memories[idx].compute_priority()
                    return True

        return False

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax for computing confidence."""
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x)

    # ── Retrieval Methods ──────────────────────────────────────────────

    def get_top_mistakes(self, n: int = 32) -> List[MistakeRecord]:
        """Get the N highest-priority mistakes for replay training."""
        sorted_mistakes = sorted(self._memories, key=lambda m: m.priority, reverse=True)
        return sorted_mistakes[:n]

    def get_mistakes_as_batch(self, n: int = 32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get top-N mistakes formatted as a training batch (inputs, targets).
        Ready to be fed directly into the network for mistake replay.
        """
        mistakes = self.get_top_mistakes(n)
        if not mistakes:
            return None

        inputs = np.array([m.input_data for m in mistakes])
        targets = np.array([m.correct_class for m in mistakes])

        return inputs, targets

    def get_confusion_pairs(self) -> Dict[Tuple[int, int], int]:
        """
        Get count of mistakes per confusion pair.
        Returns: {(correct_class, predicted_class): count}
        """
        return {k: len(v) for k, v in self._confusion_index.items() if len(v) > 0}

    def get_hardest_classes(self, top_n: int = 5) -> List[Tuple[int, int]]:
        """Get the classes with most mistakes: [(class_id, mistake_count), ...]"""
        class_mistakes = defaultdict(int)
        for m in self._memories:
            class_mistakes[m.correct_class] += 1

        sorted_classes = sorted(class_mistakes.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[:top_n]

    def sample_random(self, n: int = 32) -> List[MistakeRecord]:
        """Sample N random mistakes (useful for diverse replay)."""
        if not self._memories:
            return []
        indices = np.random.choice(len(self._memories), size=min(n, len(self._memories)), replace=False)
        return [self._memories[i] for i in indices]

    def sample_weighted(self, n: int = 32) -> List[MistakeRecord]:
        """
        Sample mistakes with probability proportional to priority.
        Higher-priority mistakes are more likely to be sampled.
        """
        if not self._memories:
            return []

        priorities = np.array([m.priority for m in self._memories])
        if priorities.sum() == 0:
            return self.sample_random(n)

        probs = priorities / priorities.sum()
        indices = np.random.choice(
            len(self._memories),
            size=min(n, len(self._memories)),
            replace=False,
            p=probs
        )
        return [self._memories[i] for i in indices]

    # ── Update Methods ─────────────────────────────────────────────────

    def mark_corrected(self, mistake_ids: List[int]):
        """Mark mistakes as corrected (model now gets them right)."""
        id_set = set(mistake_ids)
        for m in self._memories:
            if m.id in id_set:
                m.is_corrected = True
                m.priority *= 0.1  # Dramatically reduce priority
                self._total_corrections += 1

    def decay_priorities(self, factor: float = 0.95):
        """
        Apply decay to all priorities.
        Called periodically to gradually forget old, less-relevant mistakes.
        """
        for m in self._memories:
            if not m.is_corrected:
                m.priority *= factor

    def cleanup_corrected(self, keep_ratio: float = 0.1):
        """
        Remove corrected mistakes, keeping a small fraction for reference.
        This prevents the memory from being dominated by already-fixed issues.
        """
        corrected = [m for m in self._memories if m.is_corrected]
        uncorrected = [m for m in self._memories if not m.is_corrected]

        # Keep only a fraction of corrected mistakes
        n_keep = int(len(corrected) * keep_ratio)
        if corrected:
            kept_corrected = sorted(corrected, key=lambda m: m.priority, reverse=True)[:n_keep]
        else:
            kept_corrected = []

        self._memories = uncorrected + kept_corrected
        self._rebuild_confusion_index()

    def update_history(self):
        """Record current state for tracking progress over time."""
        self.mistake_count_history.append(self.size)
        if self._total_mistakes_seen > 0:
            rate = self._total_corrections / self._total_mistakes_seen
        else:
            rate = 0.0
        self.correction_rate_history.append(rate)

    # ── Statistics ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the error memory."""
        if not self._memories:
            return {
                'total_stored': 0,
                'total_seen': self._total_mistakes_seen,
                'total_corrected': self._total_corrections,
            }

        priorities = [m.priority for m in self._memories]
        confidences = [m.confidence for m in self._memories]
        losses = [m.loss_value for m in self._memories]

        return {
            'total_stored': self.size,
            'total_seen': self._total_mistakes_seen,
            'total_corrected': self._total_corrections,
            'correction_rate': self._total_corrections / max(self._total_mistakes_seen, 1),
            'avg_priority': float(np.mean(priorities)),
            'max_priority': float(np.max(priorities)),
            'avg_confidence': float(np.mean(confidences)),
            'avg_loss': float(np.mean(losses)),
            'num_confusion_pairs': len([k for k, v in self._confusion_index.items() if len(v) > 0]),
            'uncorrected_count': sum(1 for m in self._memories if not m.is_corrected),
            'corrected_count': sum(1 for m in self._memories if m.is_corrected),
        }

    def __repr__(self):
        stats = self.get_stats()
        return (f"ErrorMemoryBank(stored={stats['total_stored']}/{self.capacity}, "
                f"seen={stats['total_seen']}, corrected={stats['total_corrected']})")
