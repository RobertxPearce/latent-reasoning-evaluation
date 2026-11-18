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
