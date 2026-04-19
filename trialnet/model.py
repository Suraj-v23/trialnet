"""
model.py — TrialNet Main Model

The central class that ties everything together: core neural network,
traditional backpropagation, and the novel Try-and-Learn system.

Supports three learning modes:
- 'traditional': Standard backpropagation only
- 'trial': Try-and-Learn only (no gradient descent)
- 'hybrid': Both combined (the default and recommended mode)
"""

import numpy as np
import json
import os
import time
from typing import List, Dict, Optional, Tuple, Callable

from trialnet.core.layers import DenseLayer, DropoutLayer, BatchNormLayer, Layer
from trialnet.core.activations import get_activation
from trialnet.core.losses import Loss, get_loss
from trialnet.learning.traditional import Optimizer, get_optimizer, LearningRateScheduler
from trialnet.learning.trial_learner import TrialLearner


class TrialNet:
    """
    TrialNet — A Neural Network That Learns From Its Mistakes.

    Built entirely from scratch with NumPy.

    Usage:
        model = TrialNet(learning_mode='hybrid')
        model.add_dense(784, 256, activation='relu')
        model.add_dense(256, 128, activation='relu')
        model.add_dense(128, 10, activation='softmax')
        model.compile(optimizer='adam', loss='cross_entropy', learning_rate=0.001)
        history = model.train(X_train, y_train, epochs=20, batch_size=64,
                              validation_data=(X_val, y_val))
    """

    def __init__(self, learning_mode: str = 'hybrid', name: str = 'TrialNet'):
        """
        Args:
            learning_mode: 'traditional', 'trial', or 'hybrid'
            name: Model name
        """
        if learning_mode not in ('traditional', 'trial', 'hybrid'):
            raise ValueError(f"learning_mode must be 'traditional', 'trial', or 'hybrid'. Got: {learning_mode}")

        self.name = name
        self.learning_mode = learning_mode
        self.layers: List[Layer] = []
        self._compiled = False

        # Will be set during compile()
        self.optimizer: Optional[Optimizer] = None
        self.loss_fn: Optional[Loss] = None
        self.scheduler: Optional[LearningRateScheduler] = None
        self.trial_learner: Optional[TrialLearner] = None

        # Training history
        self.history: Dict[str, List] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'trial_metrics': [],
            'epoch_time': [],
            'mistake_count': [],
            'correction_rate': [],
        }

        # Callbacks for real-time monitoring
        self._callbacks: List[Callable] = []

    # ── Architecture Building ──────────────────────────────────────────

    def add_dense(
        self,
        input_size: int,
        output_size: int,
        activation: str = 'relu',
        init_method: str = 'he',
        name: str = ''
    ) -> 'TrialNet':
        """Add a dense (fully connected) layer."""
        layer = DenseLayer(input_size, output_size, activation=activation,
                          init_method=init_method, name=name)
        self.layers.append(layer)
        return self

    def add_dropout(self, rate: float = 0.5) -> 'TrialNet':
        """Add a dropout layer for regularization."""
        self.layers.append(DropoutLayer(rate=rate))
        return self

    def add_batchnorm(self, num_features: int) -> 'TrialNet':
        """Add batch normalization."""
        self.layers.append(BatchNormLayer(num_features=num_features))
        return self

    # ── Compilation ────────────────────────────────────────────────────

    def compile(
        self,
        optimizer: str = 'adam',
        loss: str = 'cross_entropy',
        learning_rate: float = 0.001,
        lr_schedule: str = 'warmup',
        num_classes: int = 10,
        total_epochs: int = 20,
        **optimizer_kwargs
    ):
        """
        Compile the model — set optimizer, loss, and initialize the Trial Learner.
        """
        # Collect all trainable parameters
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.get_params())

        # Set up optimizer
        self.optimizer = get_optimizer(optimizer, all_params,
                                       learning_rate=learning_rate, **optimizer_kwargs)

        # Set up learning rate scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            strategy=lr_schedule,
            total_epochs=total_epochs,
            warmup_epochs=max(2, total_epochs // 10),
        )

        # Set up loss function
        self.loss_fn = get_loss(loss)

        # Set up Trial Learner (if using trial or hybrid mode)
        if self.learning_mode in ('trial', 'hybrid'):
            self.trial_learner = TrialLearner(
                num_classes=num_classes,
                memory_capacity=2000,
                replay_batch_size=16,
                analyze_every=50,
                explore_every=100,
                trial_weight=0.05 if self.learning_mode == 'hybrid' else 0.3,
                perturbation_scale=0.001,
                n_perturbation_candidates=3,
            )
            # Trial learning rate — much smaller than main LR to prevent instability
            self._trial_lr = learning_rate * 0.05

        self._compiled = True
        self._num_classes = num_classes

        # Print model summary
        self._print_summary()

    def _print_summary(self):
        """Print model architecture summary."""
        print(f"\n{'='*60}")
        print(f"  {self.name} — Model Summary")
        print(f"  Learning Mode: {self.learning_mode.upper()}")
        print(f"{'='*60}")
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = sum(p.data.size for p in layer.get_params())
            total_params += params
            print(f"  [{i}] {layer}")
            if params > 0:
                print(f"      Parameters: {params:,}")
        print(f"{'─'*60}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Loss: {self.loss_fn.__class__.__name__}")
        if self.trial_learner:
            print(f"  Trial Learner: ENABLED")
            print(f"    Memory Capacity: {self.trial_learner.memory_bank.capacity}")
            print(f"    Trial Weight: {self.trial_learner.trial_weight}")
        print(f"{'='*60}\n")

    # ── Forward Pass ───────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass through all layers."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions (eval mode — no dropout)."""
        self._set_mode('eval')
        output = self.forward(x)
        self._set_mode('train')
        return output

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        probs = self.predict(x)
        return np.argmax(probs, axis=1)

    # ── Backward Pass ──────────────────────────────────────────────────

    def backward(self, grad_output: np.ndarray):
        """Run backward pass through all layers (reverse order)."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ── Training ───────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
        metrics_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Train the model using the Dual Learning Engine.

        Args:
            X_train: Training inputs (n_samples, n_features)
            y_train: Training targets (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional (X_val, y_val) tuple
            verbose: 0=silent, 1=progress bar, 2=detailed
            metrics_callback: Called after each epoch with metrics dict

        Returns:
            Training history dictionary
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call model.compile() first.")

        n_samples = X_train.shape[0]
        n_batches = max(n_samples // batch_size, 1)

        self._set_mode('train')

        print(f"\n🚀 Training {self.name} on {n_samples} samples...")
        print(f"   Mode: {self.learning_mode} | Epochs: {epochs} | Batch: {batch_size}\n")

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            trial_metrics_epoch = []

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # ── PATH A: Traditional Learning ───────────────────
                if self.learning_mode in ('traditional', 'hybrid'):
                    self.optimizer.zero_grad()

                    # Forward
                    predictions = self.forward(X_batch)

                    # Loss
                    loss = self.loss_fn(predictions, y_batch)
                    epoch_loss += loss

                    # Backward
                    grad = self.loss_fn.backward(predictions, y_batch)
                    self.backward(grad)

                    # Update weights
                    self.optimizer.step()
                else:
                    # Trial-only mode: still need forward pass
                    predictions = self.forward(X_batch)
                    loss = self.loss_fn(predictions, y_batch)
                    epoch_loss += loss

                # Track accuracy
                pred_classes = np.argmax(predictions, axis=1)
                if y_batch.ndim == 1:
                    true_classes = y_batch.astype(int)
                else:
                    true_classes = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(pred_classes == true_classes)
                epoch_total += len(y_batch)

                # ── PATH B: Try-and-Learn ──────────────────────────
                # Delay trial learning until epoch 4 to let the model learn basics first
                if (self.learning_mode in ('trial', 'hybrid') and 
                    self.trial_learner and epoch >= 3):
                    trial_result = self.trial_learner.step(
                        layers=self.layers,
                        loss_fn=self.loss_fn,
                        inputs=X_batch,
                        predictions=predictions,
                        targets=y_batch,
                        forward_fn=self.forward,
                        backward_fn=self._trial_backward_step,
                    )
                    trial_metrics_epoch.append(trial_result)

            # ── Epoch End ──────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / n_batches
            accuracy = epoch_correct / epoch_total

            # Update learning rate
            if self.scheduler:
                self.scheduler.step(epoch)

            # Validation
            val_loss, val_acc = 0.0, 0.0
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1])

            # Record history
            self.history['train_loss'].append(float(avg_loss))
            self.history['train_accuracy'].append(float(accuracy))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_accuracy'].append(float(val_acc))
            self.history['learning_rate'].append(float(self.scheduler.get_lr()) if self.scheduler else 0)
            self.history['epoch_time'].append(float(epoch_time))

            # Trial-specific metrics
            if self.trial_learner:
                stats = self.trial_learner.get_comprehensive_stats()
                self.history['trial_metrics'].append(stats)
                self.history['mistake_count'].append(stats['memory']['total_stored'])
                self.history['correction_rate'].append(stats['memory'].get('correction_rate', 0))
            else:
                self.history['trial_metrics'].append({})
                self.history['mistake_count'].append(0)
                self.history['correction_rate'].append(0)

            # Print progress
            if verbose >= 1:
                self._print_epoch(epoch, epochs, avg_loss, accuracy, val_loss, val_acc, epoch_time)

            if verbose >= 2 and self.trial_learner:
                self._print_trial_details()

            # Callback
            if metrics_callback:
                metrics_callback({
                    'epoch': epoch,
                    'train_loss': float(avg_loss),
                    'train_accuracy': float(accuracy),
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_acc),
                    'trial_stats': self.trial_learner.get_comprehensive_stats() if self.trial_learner else {},
                })

        print(f"\n✅ Training complete!")
        if self.trial_learner:
            final_stats = self.trial_learner.get_comprehensive_stats()
            print(f"   📊 Final mistake memory: {final_stats['memory']['total_stored']} stored, "
                  f"{final_stats['memory']['total_corrected']} corrected")
            print(f"   🎯 Correction rate: {final_stats['memory'].get('correction_rate', 0):.1%}")
            print(f"   🔬 Exploration success: {final_stats['exploration']['success_rate']:.1%}")

        return self.history

    def _trial_backward_step(self, grad_output: np.ndarray):
        """
        Backward pass for trial replay.
        
        Uses direct SGD-style updates with a small learning rate instead of
        sharing the Adam optimizer (which would corrupt its momentum state).
        """
        self.backward(grad_output)
        
        # Apply trial gradients directly with small LR (no momentum)
        # This prevents corrupting the main Adam optimizer's state
        trial_lr = getattr(self, '_trial_lr', 0.0001)
        for layer in self.layers:
            for param in layer.get_params():
                if param.grad is not None:
                    # Direct SGD update with gradient clipping
                    clipped_grad = np.clip(param.grad, -1.0, 1.0)
                    param.data -= trial_lr * clipped_grad
                    param.zero_grad()

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on data. Returns (loss, accuracy)."""
        self._set_mode('eval')
        predictions = self.forward(X)
        loss = float(self.loss_fn(predictions, y))

        pred_classes = np.argmax(predictions, axis=1)
        if y.ndim == 1:
            true_classes = y.astype(int)
        else:
            true_classes = np.argmax(y, axis=1)

        accuracy = float(np.mean(pred_classes == true_classes))
        self._set_mode('train')

        return loss, accuracy

    # ── Save/Load ──────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model weights and configuration."""
        os.makedirs(path, exist_ok=True)

        # Save weights
        weights = {}
        for i, layer in enumerate(self.layers):
            for j, param in enumerate(layer.get_params()):
                weights[f"layer_{i}_param_{j}"] = param.data

        np.savez(os.path.join(path, 'weights.npz'), **weights)

        # Save config
        config = {
            'name': self.name,
            'learning_mode': self.learning_mode,
            'num_layers': len(self.layers),
            'layer_configs': [],
        }
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                config['layer_configs'].append({
                    'type': 'dense',
                    'input_size': layer.input_size,
                    'output_size': layer.output_size,
                    'activation': layer.activation_name,
                })
            elif isinstance(layer, DropoutLayer):
                config['layer_configs'].append({
                    'type': 'dropout',
                    'rate': layer.rate,
                })
            elif isinstance(layer, BatchNormLayer):
                config['layer_configs'].append({
                    'type': 'batchnorm',
                    'num_features': layer.num_features,
                })

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Save history
        serializable_history = {}
        for k, v in self.history.items():
            if k == 'trial_metrics':
                serializable_history[k] = [self._serialize_metrics(m) for m in v]
            else:
                serializable_history[k] = v

        with open(os.path.join(path, 'history.json'), 'w') as f:
            json.dump(serializable_history, f, indent=2)

        print(f"💾 Model saved to {path}")

    def _serialize_metrics(self, metrics: Dict) -> Dict:
        """Convert metrics dict to JSON-serializable format."""
        result = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                result[k] = self._serialize_metrics(v)
            elif isinstance(v, (np.integer, np.floating)):
                result[k] = float(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                result[k] = [self._serialize_item(item) for item in v]
            else:
                result[k] = v
        return result

    def _serialize_item(self, item):
        """Serialize a single item."""
        if isinstance(item, (np.integer, np.floating)):
            return float(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, dict):
            return self._serialize_metrics(item)
        if isinstance(item, (list, tuple)):
            return [self._serialize_item(i) for i in item]
        return item

    @classmethod
    def load(cls, path: str) -> 'TrialNet':
        """Load model from saved files."""
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)

        model = cls(learning_mode=config['learning_mode'], name=config['name'])

        # Reconstruct architecture
        for layer_config in config['layer_configs']:
            if layer_config['type'] == 'dense':
                model.add_dense(
                    layer_config['input_size'],
                    layer_config['output_size'],
                    activation=layer_config['activation'],
                )
            elif layer_config['type'] == 'dropout':
                model.add_dropout(rate=layer_config['rate'])
            elif layer_config['type'] == 'batchnorm':
                model.add_batchnorm(num_features=layer_config['num_features'])

        # Load weights
        weights = np.load(os.path.join(path, 'weights.npz'))
        for i, layer in enumerate(model.layers):
            for j, param in enumerate(layer.get_params()):
                key = f"layer_{i}_param_{j}"
                if key in weights:
                    param.data = weights[key]

        # Load history
        history_path = os.path.join(path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                model.history = json.load(f)

        print(f"📂 Model loaded from {path}")
        return model

    # ── Utility ────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        """Set training or eval mode for all layers."""
        for layer in self.layers:
            if mode == 'train':
                layer.train()
            else:
                layer.eval()

    def _print_epoch(self, epoch, total_epochs, loss, acc, val_loss, val_acc, time_s):
        """Print epoch progress."""
        bar_width = 20
        progress = (epoch + 1) / total_epochs
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)

        line = (f"  Epoch {epoch+1:3d}/{total_epochs} [{bar}] "
                f"loss: {loss:.4f} acc: {acc:.4f}")

        if val_loss > 0:
            line += f" | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"

        line += f" ({time_s:.1f}s)"

        # Trial learner status
        if self.trial_learner:
            mem_size = self.trial_learner.memory_bank.size
            line += f" | 🧠 mem: {mem_size}"

        print(line)

    def _print_trial_details(self):
        """Print detailed trial learner information."""
        if not self.trial_learner:
            return

        stats = self.trial_learner.get_comprehensive_stats()
        print(f"       ├─ Memory: {stats['memory']['total_stored']} stored, "
              f"{stats['memory']['total_corrected']} corrected")
        print(f"       ├─ Explorer: {stats['exploration']['success_rate']:.1%} success rate")
        if stats.get('latest_report'):
            report = stats['latest_report']
            print(f"       └─ Analysis: severity={report['severity']:.2f}, "
                  f"patterns={report['patterns']}, "
                  f"hardest={report['hardest_classes']}")

    def get_mistake_report(self):
        """Get the latest mistake analysis report."""
        if self.trial_learner:
            return self.trial_learner.get_mistake_report()
        return None

    def __repr__(self):
        n_params = sum(p.data.size for layer in self.layers for p in layer.get_params())
        return f"TrialNet(name='{self.name}', mode='{self.learning_mode}', layers={len(self.layers)}, params={n_params:,})"
