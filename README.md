# RL-Portfolio-Allocator

A portfolio allocation system using Reinforcement Learning Techniques

### Available RL Models
- Advantage Actor Critic (A2C)
- Deep Deterministic Policy Gradient (DDPG)
- Soft Actor Critic (SAC)
- Proximal Policy Optimization (PPO)

### Training
To train a model, use [train.py](train.py) python script.

``` bash
python train.py <model_name> <start_date> <end_date> [OPTIONAL - initial_amount]
```

> [!NOTE]
> - For training, use any one of the available models (a2c, ddpg, sac, ppo).

The pretrained trained models are stored in [portfolio_manager/trained_models](portfolio_manager/trained_models).
