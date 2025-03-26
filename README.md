# THESIS CODE
This repository contains a collection of MATLAB and Python scripts for EPANET-based dataset generation, neural network tuning (network structure + regularization), and sensor pruning based on learned weights. The workflow is divided into steps, each with its own script/notebook, as discussed in my thesis[1] and in [2].
The workflow is divided into steps, each with its own script/notebook. Below is an overview of how to set up and run them.

# Scripts Overview
1. Dataset Generation & Splitting
> **Note:** The data used in this repository cannot be publicly shared due to confidentiality constraints. However, the same methodology applies to *any* Water Distribution Network. By simulating an EPANET model (see [1,2] for guidelines), you can generate similar data to train and test your models.

This step involves:
- Generating synthetic or real data from an EPANET simulation.
- Organizing the resulting data into Train, Validation, and Test sets.

2. Network Structure Tuning

    tuning_net_structure.py
    Uses Python’s Keras Tuner to optimize neural net architecture (e.g., layer sizes, learning rate) with Val1.
    The results, including logs and hyperparameter grids, are saved in INFO/ subfolders.

3. L21 Regularization Tuning

    tuning_L21.py
    Focuses on tuning the L21 (group-sparse) regularization hyperparameter. Trains on Train+Val1 and checks performance on Val2.

4. Pruning Sensors via Norm2 & Retraining

Two key MATLAB scripts plus corresponding Python evaluation scripts:

    allModelsThAndGridEvaluation.m
    Applies norm-2 thresholds to input weights to decide which sensors to keep.

    evaluateModelAtDifferentTh.py
    A Python function invoked by the MATLAB script to evaluate the model across thresholds.

You only need to do extensive pruning for the best model. Use plotL21.m to find the final threshold.

5. Final Single-Threshold Pruning

    allModelsThAndGridEvaluation_singleTh.m
    Once the threshold is fixed, re-train from optimized weights, again using Train+Val1 / Val2.

    evaluateModelAtSingleTh.py
    Python script to evaluate or finalize results.

## References
[1] **Lo Presti, J.**. "Neural Network-Based Methods for the Management and Control of Complex Systems." (2025).

[2] **Lo Presti, J., et al.** (2024). *Combining clustering and regularised neural network for burst detection and localization and flow/pressure sensor placement in water distribution networks.* Journal of Water Process Engineering, **63**, 105473.
