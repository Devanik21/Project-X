# Project X

![Language](https://img.shields.io/badge/Language-Jupyter%20Notebook-DA5B0B?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Project-X?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Project-X?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> An experimental research sandbox — exploring the intersection of novel AI architectures, unconventional problem formulations, and frontier research ideas.

---

**Topics:** `aion` · `cognitive-architecture` · `genomic-error-correction` · `hrf-titan-26` · `longevity-ai` · `nanoscale-digital-organisms` · `physics-informed-architectures` · `resilient-intelligence` · `universal-signal-reasoning` · `bshder`

## Overview

Project-X is a research sandbox repository for experimental AI work that does not yet fit neatly into
a defined product category. It contains a collection of exploratory implementations, prototype
architectures, and research experiments at various stages of development — from validated ideas ready
for formal write-up to speculative implementations testing hypotheses that may or may not pan out.

The philosophy of Project-X is to maintain a disciplined record of research exploration: every
experiment is documented with its hypothesis, implementation, results (positive or negative), and
interpretation. Failed experiments are retained and documented rather than discarded — they constitute
negative results that are scientifically valuable and prevent duplicate effort. This reflects a mature
research practice where the exploration process is as important as the final polished result.

Current active threads include: novel loss functions for structured prediction problems, architectural
experiments with hybrid recurrent-attention models, and exploration of physics-inspired regularisation
techniques for neural network training. Each thread is isolated in its own subdirectory with a
dedicated README documenting the specific hypothesis, methodology, and current status.

---

## Motivation

Good research requires space to explore bad ideas before finding good ones. Project-X exists because
not every interesting idea deserves its own repository from day one — some ideas need to be
implemented, evaluated, and either validated for promotion to a proper project or documented as
informative negative results. This repository provides that space without the overhead of
formal project structure.

---

## Architecture

```
Project-X Research Repository
        │
  ┌─────────────────────────────────────────────┐
  │  /experiments/                              │
  │  ├── exp_001_physics_regularisation/        │
  │  │   ├── README.md (hypothesis + results)   │
  │  │   ├── model.py                           │
  │  │   └── results/                           │
  │  ├── exp_002_hybrid_attention/              │
  │  ├── exp_003_structured_loss/               │
  │  └── ...                                    │
  ├── /shared/                                  │
  │  ├── utils.py (shared experiment utilities) │
  │  └── datasets.py (shared data loaders)     │
  └── /archive/ (documented negative results)  │
  └─────────────────────────────────────────────┘
```

---

## Features

### Experiment Documentation Framework
Each experiment follows a mandatory documentation template: hypothesis, methodology, implementation notes, results (quantitative and qualitative), and conclusion — creating a searchable research log.

### Isolated Experiment Directories
Each experimental thread isolated in its own directory with independent dependencies, preventing cross-contamination between experiments at different stages of development.

### Negative Result Archive
A dedicated archive directory for documented failed experiments — maintaining negative results with their interpretation as a form of institutional knowledge.

### Shared Utilities Layer
Common experiment infrastructure in /shared/: data loaders, evaluation harnesses, visualisation utilities, and experiment tracking setup — preventing duplicate boilerplate across experiments.

### Hypothesis Tracking
Structured hypothesis statements in each experiment README linking the motivation, the specific claim being tested, and the evaluation criteria that would confirm or refute it.

### Reproducibility Contracts
Each experiment specifies its exact seed, dataset version, hardware configuration, and dependency versions — making results reproducible by others or by the author six months later.

### Experiment Status Tags
Status labels per experiment: `HYPOTHESIS`, `IN_PROGRESS`, `VALIDATED`, `NEGATIVE_RESULT`, `PROMOTED` (moved to standalone repo) — providing immediate orientation to the repository state.

### Cross-Experiment References
Explicit links between experiments that build on each other, were inspired by each other, or contradict each other — maintaining the intellectual lineage of the research.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **PyTorch** | Deep learning framework | Neural network experiments and training loops |
| **Weights & Biases / MLflow** | Experiment tracking | Run logging, metric visualisation, hyperparameter comparison |
| **pytest** | Testing | Unit tests for shared utilities and experiment components |
| **Hydra / YAML** | Configuration | Structured experiment configuration management |
| **pandas** | Results analysis | Experiment result aggregation and comparison |
| **Matplotlib / Plotly** | Visualisation | Result plots for each experiment |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Project-X.git
cd Project-X
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run a specific experiment
cd experiments/exp_001_physics_regularisation/
python run_experiment.py --config config.yaml

# Run all experiments
python run_all_experiments.py --skip_archived True
```

---

## Usage

```bash
# Run experiment 001
python experiments/exp_001/run.py --seed 42

# View experiment results
python summarise_results.py --experiment exp_001

# Add a new experiment
python new_experiment.py --name 'adaptive_lr_physics' --hypothesis 'Physics-informed LR schedule reduces loss landscape roughness'

# Archive a failed experiment
python archive_experiment.py --experiment exp_002 --reason 'No improvement over baseline on 3 datasets'
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `EXPERIMENT_DIR` | `experiments/` | Root directory for experiment subdirectories |
| `TRACKING_BACKEND` | `wandb` | Experiment tracking: wandb, mlflow, tensorboard, none |
| `DEFAULT_SEED` | `42` | Default random seed for reproducibility |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Project-X/
├── README.md
├── 01_Foundations/crude_chess_rl.py
├── 01_Foundations/evolving_canvas.py
├── 02_Sophisticated_Architectures/the_dark_lucid_dream.py
├── 04_Dark_Lucid_Dream/the_dark_lucid_dream.py
├── 01_Foundations/6-powerful-reinforcement-learning-algorithms.ipynb
├── 01_Foundations/checkers-rl.ipynb
├── 01_Foundations/deep-reinforcement-learning (1).ipynb
└── ...
```

---

## Roadmap

- [ ] Automated experiment comparison report generation across all validated experiments
- [ ] Continuous integration: run lightweight experiment smoke tests on every commit
- [ ] Experiment promotion pipeline: automated scaffolding to promote validated experiments to standalone repos
- [ ] Research journal integration: link experiments to the specific research question they address
- [ ] Collaborative mode: shared experiment registry for team research with ownership tracking

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

This is a research sandbox — code quality and testing standards are intentionally lower than production repositories. Experiments marked IN_PROGRESS or HYPOTHESIS may not run without modification. Only experiments marked VALIDATED are considered reliable results.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
