# An Investigation of Latent Reasoning's Generalizability and Calibration

Johns Hopkins University  
John Guerrerio and Robert Pearce

## Overview

Latent reasoning lets an LLM "think" in its continuous hidden states rather than in an
explicit textual chain of thought (CoT). This can improve both performance and token
efficiency. Most work so far has focused on making these systems perform well. Much less
attention has gone to how well they **generalize** or how well **calibrated** they are.

This project evaluates those two properties in models trained with two foundational
latent reasoning methods, **COCONUT** and **CODI**. Each is compared against a textual-CoT
model and a no-CoT model. We then aim to use the findings for OOD detection, abstention,
and a variable number of latent thoughts.

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/RobertxPearce/latent-reasoning-evaluation.git

# Create an environment and install our package (editable) + dev tools
conda create -n lre python=3.12 -y
conda activate lre
pip install -e ".[dev]"
```

## Code Organization

Shared code lives in the `lre` package under `src/lre/`. Because it is installed in
editable mode, scripts and notebooks import it directly (e.g.
`from lre.metrics import accuracy`), and changes to the package take effect immediately.

```
src/lre/
├── data/        # dataset loading, easy/hard splits, COCONUT and CODI formats
└── metrics/     # answer extraction and accuracy
tests/           # tests for the package, run with pytest
scripts/         # data preparation, training, and evaluation
notebooks/       # analysis and figures
third_party/     # COCONUT, CODI, and ReProbe as git submodules
```

## Acknowledgments and Licenses

This repository references third-party code as git submodules.
- COCONUT: Hao et al., *Training Large Language Models to Reason in a Continuous Latent
  Space*. MIT License.
- CODI: Shen et al., *CODI: Compressing Chain-of-Thought into Continuous Space via
  Self-Distillation*. No license file in the upstream repository.
- ReProbe: Ni et al., *ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by
  Probing Internal States of Large Language Models*. No license file in the upstream
  repository.
- Easy2Hard-Bench: Ding et al., *Easy2Hard-Bench: Standardized Difficulty Labels for
  Profiling LLM Performance and Generalization*.
- GSM8K: Cobbe et al., *Training Verifiers to Solve Math Word Problems*.
