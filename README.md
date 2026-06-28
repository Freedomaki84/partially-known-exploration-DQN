# partially-known-exploration-DQN
Memory-Efficient Exploration in Partially Known Environments (DQN)
This repository contains the implementation, training pipelines, and evaluation framework for the Master’s Thesis: "Memory-Efficient Exploration in Partially Known Environments using Deep Q-Networks (DQN)" (November 2025).

Overview
This research investigates how different Deep Q-Network (DQN) variants balance exploration performance and memory efficiency in partially observable environments. Using the MiniGrid-DoorKey-5×5 task as a benchmark, this project provides a comparative analysis of three DQN architectures under varying replay-buffer sizes (25K, 50K, and 100K).

Research Focus
Balancing exploration and memory: Evaluating whether algorithmic refinement is more effective than simply increasing replay buffer capacity.

Algorithms implemented:

Vanilla DQN: Canonical baseline for off-policy value-based learning.

Double DQN: Mitigates overestimation bias through decoupled value evaluation.

β-DQN: Exploration-focused extension using auxiliary statistics to quantify action frequency.

Key Findings
Double DQN achieves superior task success with the smallest memory footprint, showing remarkable robustness across different buffer sizes.

Vanilla DQN shows a higher dependence on buffer capacity to reach stable convergence.

β-DQN demonstrates the potential of behaviour-driven exploration, though it proved sensitive to hyperparameters under strict masking.

The study concludes that exploration quality (strategic state coverage) is more critical than exploration quantity (larger replay buffers) for resource-constrained reinforcement learning.

1. Setup and Installation
Clone the repository:
```
git clone https://github.com/Freedomaki84/partially-known-exploration-DQN.git
cd partially-known-exploration-DQN
```

3. Setup virtual environment:
```
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

Usage
Training: Execute the training scripts to reproduce the experiments:
```
python train_5x5_dqn.py  # Example command
```
Evaluation: Use the provided monitoring tools to visualize training stability and memory usage via TensorBoard logs in tb_logs/.

Citation
If you find this research or codebase useful in your own work, please cite the thesis:

Pavlatou, E. (2025). Memory-Efficient Exploration in Partially Known Environments using Deep Q-Networks (DQN). Master Thesis, Program of Study: Master in AI.

