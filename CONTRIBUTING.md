# Contributing to TrialNet

Thank you for your interest in contributing to TrialNet! 🎉

This document outlines how you can help and what to expect from the contribution process.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/trialnet.git
   cd trialnet
   ```
3. **Set up** the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Create a **feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Types of Contributions

### 🐛 Bug Reports
- Open an issue using the **Bug Report** template
- Include steps to reproduce, expected vs. actual behaviour, and your environment details

### 💡 Feature Requests
- Open an issue using the **Feature Request** template
- Check the [ROADMAP.md](ROADMAP.md) first to see if it's already planned

### 🔧 Pull Requests
- Keep changes focused — one feature/fix per PR
- Write clear commit messages
- Update documentation if needed
- Test your changes before submitting

## 🎯 Priority Areas

Looking for places to start? Here are high-impact areas:
- **New learning algorithms** — alternatives to perturbation exploration
- **Dataset support** — beyond MNIST (CIFAR-10, custom datasets)
- **Better judge prompts** — improve the LLM-as-Judge accuracy
- **Tests** — unit tests for the core NumPy layers
- **Documentation** — tutorials, examples, diagrams

## 📝 Code Style

- Follow **PEP 8** for Python code
- Add **docstrings** to all classes and public methods
- Keep functions focused and small

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
