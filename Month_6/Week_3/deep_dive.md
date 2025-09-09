# Month 6, Week 3: Deep Dive - Deep Reinforcement Learning

## Deep Q-Networks (DQN)
- **Concept**: Combines Q-learning with deep neural networks to handle high-dimensional state spaces.
- **Neural Network as Q-function Approximator**: The neural network takes the state as input and outputs Q-values for all possible actions.
- **Experience Replay**: Stores past experiences (S, A, R, S') in a replay buffer and samples mini-batches for training. Breaks correlations in the data and improves stability.
- **Target Network**: Uses a separate, delayed-update target network for calculating the TD target. This stabilizes the training process.
- **DQN Variants**: Double DQN (addresses overestimation of Q-values), Dueling DQN (separates state-value and advantage functions), Prioritized Experience Replay (samples important experiences more frequently).

## Policy Gradient Methods
- **Concept**: Directly learn a parameterized policy $\pi_{\theta}(a|s)$ that maps states to actions, without explicitly learning a value function.
- **Objective**: Maximize the expected return $J(\theta) = \mathbb{E}[G_t]$.
- **Policy Gradient Theorem**: Provides a way to compute the gradient of the expected return with respect to the policy parameters $\theta$.
- **REINFORCE (Monte Carlo Policy Gradient)**: An episodic policy gradient algorithm that uses the return from an entire episode to update the policy.

## Actor-Critic Methods
- **Concept**: Combine policy gradient (Actor) with value-based methods (Critic).
- **Actor**: The policy network, responsible for selecting actions.
- **Critic**: The value network, responsible for estimating the value function (e.g., $V(s)$ or $Q(s, a)$) to guide the actor's updates.
- **Advantage Function**: Often used to reduce variance in policy gradient updates: $A(s, a) = Q(s, a) - V(s)$.

## Advanced DRL Algorithms
- **A2C (Advantage Actor-Critic)**: A synchronous, deterministic version of A3C.
- **A3C (Asynchronous Advantage Actor-Critic)**: Uses multiple agents exploring the environment in parallel and asynchronously updating a global network. Improves exploration and training stability.
- **PPO (Proximal Policy Optimization)**: A policy gradient method that uses a clipped surrogate objective function to prevent large policy updates, leading to more stable and efficient training.
- **DDPG (Deep Deterministic Policy Gradient)**: An actor-critic algorithm for continuous action spaces. Uses a deterministic policy and a Q-network, similar to DQN but for continuous actions.
