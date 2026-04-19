"""
mistake_analyzer.py — Mistake Pattern Analyzer for TrialNet 🆕 NOVEL

Analyzes the Error Memory Bank to discover PATTERNS in the model's mistakes.
Instead of treating each error independently, this module finds systemic
issues — like "the model always confuses 3 and 8" — and produces
actionable insights for the Try-and-Learn engine.

Analysis types:
1. Confusion Matrix Analysis — Which classes get confused?
2. Feature Attribution — Which input features cause problems?
3. Confidence Calibration — Is the model overconfident when wrong?
4. Temporal Patterns — Are mistakes getting worse or better?
5. Cluster Analysis — Group mistakes into categories
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from trialnet.learning.error_memory import ErrorMemoryBank, MistakeRecord


@dataclass
class MistakePattern:
    """A discovered pattern in the model's mistakes."""
    pattern_type: str  # 'confusion', 'feature_blind_spot', 'overconfidence', etc.
    description: str
    severity: float  # 0-1 scale
    affected_classes: List[int]
    affected_features: Optional[np.ndarray] = None
    recommendation: str = ""
    sample_count: int = 0


@dataclass
class MistakeReport:
    """
    Complete analysis of the model's mistakes.
    Produced by the Mistake Pattern Analyzer and consumed by the Trial Learner.
    """
    # Discovered patterns
    patterns: List[MistakePattern] = field(default_factory=list)

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Top confused class pairs: [(class_a, class_b, count), ...]
    top_confusions: List[Tuple[int, int, int]] = field(default_factory=list)

    # Classes the model struggles with most
    hardest_classes: List[int] = field(default_factory=list)

    # Feature importance for mistakes (which input features cause errors)
    mistake_feature_importance: Optional[np.ndarray] = None

    # Confidence calibration: (avg_confidence_when_wrong, avg_confidence_when_right)
    confidence_calibration: Tuple[float, float] = (0.0, 0.0)

    # Overall mistake severity score (0-1)
    overall_severity: float = 0.0

    # Recommended training focus weights per class
    class_focus_weights: Optional[np.ndarray] = None

    # Number of mistakes analyzed
    total_analyzed: int = 0

    # Recommendations for the trial learner
    focus_on_classes: List[int] = field(default_factory=list)
    perturbation_strength: float = 0.01
    replay_ratio: float = 0.3


class MistakePatternAnalyzer:
    """
    Analyzes the Error Memory Bank to find systemic patterns in mistakes.

    This is the "intelligence" behind the Try-and-Learn engine — it tells
    the Perturbation Explorer and Trial Learner WHERE to focus their efforts.
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self._report_history: List[MistakeReport] = []

    def analyze(
        self,
        memory_bank: ErrorMemoryBank,
        recent_accuracy: float = 0.0
    ) -> MistakeReport:
        """
        Run a full analysis of the Error Memory Bank.
        Returns a MistakeReport with patterns and recommendations.
        """
        report = MistakeReport()
        report.total_analyzed = memory_bank.size

        if memory_bank.size == 0:
            return report

        mistakes = memory_bank._memories

        # Run all analyses
        report.confusion_matrix = self._build_confusion_matrix(mistakes)
        report.top_confusions = self._find_top_confusions(report.confusion_matrix)
        report.hardest_classes = self._find_hardest_classes(mistakes)
        report.mistake_feature_importance = self._analyze_feature_importance(mistakes)
        report.confidence_calibration = self._analyze_confidence(mistakes)
        report.patterns = self._detect_patterns(mistakes, report)
        report.overall_severity = self._compute_severity(mistakes, recent_accuracy)
        report.class_focus_weights = self._compute_focus_weights(mistakes)

        # Generate recommendations
        self._generate_recommendations(report)

        # Save history
        self._report_history.append(report)

        return report

    def _build_confusion_matrix(self, mistakes: List[MistakeRecord]) -> np.ndarray:
        """
        Build a confusion matrix from mistakes.
        confusion[i][j] = number of times class i was predicted as class j.
        """
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for m in mistakes:
            if m.correct_class < self.num_classes and m.predicted_class < self.num_classes:
                matrix[m.correct_class][m.predicted_class] += 1
        return matrix

    def _find_top_confusions(self, confusion_matrix: np.ndarray, top_n: int = 10) -> List[Tuple[int, int, int]]:
        """Find the most common confusion pairs."""
        confusions = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and confusion_matrix[i][j] > 0:
                    confusions.append((i, j, int(confusion_matrix[i][j])))

        confusions.sort(key=lambda x: x[2], reverse=True)
        return confusions[:top_n]

    def _find_hardest_classes(self, mistakes: List[MistakeRecord]) -> List[int]:
        """Find classes with the most mistakes."""
        class_counts = defaultdict(int)
        for m in mistakes:
            class_counts[m.correct_class] += 1

        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_classes[:5]]

    def _analyze_feature_importance(self, mistakes: List[MistakeRecord]) -> Optional[np.ndarray]:
        """
        Analyze which input features are most associated with mistakes.

        Uses variance analysis: features with high variance across mistakes
        but consistent patterns within confusion pairs might be the "blind spots"
        that cause confusion.
        """
        if not mistakes:
            return None

        # Get all mistake inputs
        inputs = np.array([m.input_data.flatten() for m in mistakes])
        n_features = inputs.shape[1]

        # Compute per-feature statistics across mistakes
        feature_variance = np.var(inputs, axis=0)
        feature_mean = np.mean(np.abs(inputs), axis=0)

        # Features with low activation in mistakes might be "ignored" features
        # that could help distinguish confused classes
        importance = feature_variance * feature_mean

        # Normalize
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    def _analyze_confidence(self, mistakes: List[MistakeRecord]) -> Tuple[float, float]:
        """
        Analyze confidence calibration.

        Returns (avg_confidence_when_wrong, ideal_confidence).
        A well-calibrated model should have LOW confidence when wrong.
        """
        confidences = [m.confidence for m in mistakes]
        avg_wrong_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Ideal: confidence when wrong should be close to 1/num_classes (random)
        ideal_confidence = 1.0 / self.num_classes

        return (avg_wrong_confidence, ideal_confidence)

    def _detect_patterns(
        self,
        mistakes: List[MistakeRecord],
        report: MistakeReport
    ) -> List[MistakePattern]:
        """Detect specific patterns in mistakes."""
        patterns = []

        # Pattern 1: Systematic confusion pairs
        for correct, predicted, count in report.top_confusions[:5]:
            if count >= 3:
                patterns.append(MistakePattern(
                    pattern_type='confusion',
                    description=f"Systematically confuses class {correct} with class {predicted}",
                    severity=min(count / 50.0, 1.0),
                    affected_classes=[correct, predicted],
                    recommendation=f"Focus perturbation on features distinguishing class {correct} from {predicted}",
                    sample_count=count,
                ))

        # Pattern 2: Overconfidence
        avg_conf, ideal_conf = report.confidence_calibration
        if avg_conf > 0.8:
            patterns.append(MistakePattern(
                pattern_type='overconfidence',
                description=f"Model is overconfident when wrong (avg confidence: {avg_conf:.2f})",
                severity=min(avg_conf, 1.0),
                affected_classes=report.hardest_classes,
                recommendation="Apply stronger perturbation and increase dropout",
                sample_count=len(mistakes),
            ))

        # Pattern 3: Specific class failure
        class_counts = defaultdict(int)
        for m in mistakes:
            class_counts[m.correct_class] += 1
        total = sum(class_counts.values())
        for cls, count in class_counts.items():
            ratio = count / max(total, 1)
            if ratio > 0.25:  # One class has >25% of all mistakes
                patterns.append(MistakePattern(
                    pattern_type='class_failure',
                    description=f"Class {cls} accounts for {ratio:.0%} of all mistakes",
                    severity=ratio,
                    affected_classes=[cls],
                    recommendation=f"Boost training on class {cls} samples",
                    sample_count=count,
                ))

        # Pattern 4: Stubborn mistakes (many correction attempts, still wrong)
        stubborn = [m for m in mistakes if m.correction_attempts >= 3 and not m.is_corrected]
        if len(stubborn) >= 5:
            patterns.append(MistakePattern(
                pattern_type='stubborn',
                description=f"{len(stubborn)} mistakes persist despite multiple correction attempts",
                severity=min(len(stubborn) / 20.0, 1.0),
                affected_classes=list(set(m.correct_class for m in stubborn)),
                recommendation="Try larger perturbations or architectural changes for these samples",
                sample_count=len(stubborn),
            ))

        return patterns

    def _compute_severity(self, mistakes: List[MistakeRecord], accuracy: float) -> float:
        """Compute overall severity score (0-1)."""
        if not mistakes:
            return 0.0

        avg_loss = float(np.mean([m.loss_value for m in mistakes]))
        avg_confidence = float(np.mean([m.confidence for m in mistakes]))
        error_rate = 1.0 - accuracy

        # Weighted combination
        severity = (
            0.3 * min(avg_loss / 5.0, 1.0) +
            0.3 * avg_confidence +
            0.4 * error_rate
        )

        return float(np.clip(severity, 0, 1))

    def _compute_focus_weights(self, mistakes: List[MistakeRecord]) -> np.ndarray:
        """
        Compute per-class focus weights for training.

        Classes with more mistakes get higher weights, telling the
        Trial Learner to spend more time on them.
        """
        weights = np.ones(self.num_classes)
        class_counts = defaultdict(int)
        for m in mistakes:
            class_counts[m.correct_class] += 1

        total = sum(class_counts.values())
        if total > 0:
            for cls, count in class_counts.items():
                if cls < self.num_classes:
                    # Weight proportional to mistake frequency
                    weights[cls] = 1.0 + (count / total) * 5.0

        return weights

    def _generate_recommendations(self, report: MistakeReport):
        """Generate actionable recommendations for the Trial Learner."""
        # Focus on top 3 hardest classes
        report.focus_on_classes = report.hardest_classes[:3]

        # Adjust perturbation strength based on severity
        if report.overall_severity > 0.7:
            report.perturbation_strength = 0.05  # Explore more
        elif report.overall_severity > 0.3:
            report.perturbation_strength = 0.02  # Moderate exploration
        else:
            report.perturbation_strength = 0.005  # Fine-tune

        # Adjust replay ratio
        if report.patterns:
            report.replay_ratio = min(0.5, 0.1 + 0.1 * len(report.patterns))
        else:
            report.replay_ratio = 0.1

    def get_trend(self) -> Dict:
        """Analyze trends across multiple reports."""
        if len(self._report_history) < 2:
            return {'trend': 'insufficient_data'}

        recent = self._report_history[-1]
        previous = self._report_history[-2]

        severity_change = recent.overall_severity - previous.overall_severity
        pattern_change = len(recent.patterns) - len(previous.patterns)

        if severity_change < -0.05:
            trend = 'improving'
        elif severity_change > 0.05:
            trend = 'worsening'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'severity_change': float(severity_change),
            'pattern_change': pattern_change,
            'current_severity': float(recent.overall_severity),
        }
