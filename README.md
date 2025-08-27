# Asynchronous Swarm Confrontation
A Simulation Platform for MARL Training and Evaluation in Swarm Confrontation
Multi-agent reinforcement learning (MARL) provides a robust framework for tackling task and motion planning challenges, particularly in swarm confrontation scenarios.
By customizing termination conditions for diverse tasks, event-driven MARL reduces decision jitter stemming from frequent task switching.
However, it hinders robots from updating strategies on a consistent timescale, leading to misaligned information sharing that disrupts agent coordination.
To address this, we propose a novel event-driven MARL approach that facilitates collaborative strategy learning under asynchronous conditions.
We introduce an experience selection scheme tailored to diverse timescales, ensuring efficient training with synchronized information among robots.
By incorporating Transformers, our method enables robots to infer others' behaviors from historical data, optimizing collaborative strategies.
Extensive experiments validate the effectiveness of our proposed approach.


# Installation instructions:
The code has the following requirements: Python 3.8.
We include a requirements.txt file as a reference, but note that such a file includes more libraries than the ones strictly needed to run our code.

# How to run the code:
To run the code, execute the following files:
Training: python train_qmix.py
Testing: python test_qmix.py
Rendering: python train_render.py
