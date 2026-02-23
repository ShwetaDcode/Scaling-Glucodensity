# Deep Evidential Engine for Metabolic Signal Analysis

An automated, optimized numerical framework for processing high-frequency functional data (CGM) using **Deep Evidential Regression**. This system replaces traditional iterative statistical models with a differentiable PyTorch engine capable of real-time uncertainty quantification.

## The Three Pillars of the Framework

1.  **Computational Optimization:** Uses a differentiable Neural Backbone to approximate basis functions. This enables vectorized tensor operations that are significantly faster than CPU-bound iterative matrix inversions.
2.  **Pipeline Automation:** Implements a `ClinicalPipeline` wrapper using the `Scikit-Learn` API. This standardizes the ingestion of raw clinical longitudinal data directly into deep learning tensors.
3.  **Latency Reduction:** Utilizing **Normal-Inverse-Gamma (NIG)** priors, the engine extracts both Aleatoric (sensor noise) and Epistemic (data sparsity) uncertainty in a **single forward pass**, eliminating the need for slow Monte Carlo simulations.

## Installation

Ensure you have Python 3.9+ installed.

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd metabolic-evidential-engine
