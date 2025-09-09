# Month 6, Week 1: Deep Dive - Reinforcement Learning Fundamentals

## Core Concepts of Reinforcement Learning (RL)
- **Agent**: The learner or decision-maker.
- **Environment**: The world with which the agent interacts.
- **State (S)**: A complete description of the environment at a given time.
- **Action (A)**: A choice made by the agent that can change the state of the environment.
- **Reward (R)**: A scalar feedback signal from the environment, indicating the desirability of the agent's action.
- **Policy ($\pi$)**: A mapping from states to probabilities of selecting each possible action.
- **Value Function (V or Q)**: A prediction of future rewards, used to evaluate the goodness of states or state-action pairs.

## Markov Decision Processes (MDPs)
- **Definition**: A mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker.
- **Components**: (S, A, P, R, $\gamma$)
    - S: Set of states.
    - A: Set of actions.
    - P: State transition probability function $P(s'|s, a)$.
    - R: Reward function $R(s, a, s')$.
    - $\gamma$: Discount factor (0 $\le \gamma \le$ 1).

## The Bellman Equation
- **Purpose**: Provides a fundamental relationship between the value of a state and the values of its successor states. It's a necessary condition for optimality.
- **Bellman Expectation Equation (for a given policy $\pi$)**:
    - State-Value Function: $V^{\pi}(s) = \mathbb{E}_{a \sim \pi(s), s' \sim P(s'|s,a)}[R_{t+1} + \gamma V^{\pi}(s') | S_t = s]$
    - Action-Value Function: $Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P(s'|s,a)}[R_{t+1} + \gamma V^{\pi}(s') | S_t = s, A_t = a]$
- **Bellman Optimality Equation (for the optimal policy $\pi^*$)**:
    - Optimal State-Value Function: $V^*(s) = \max_a \mathbb{E}_{s' \sim P(s'|s,a)}[R_{t+1} + \gamma V^*(s') | S_t = s]$
    - Optimal Action-Value Function: $Q^*(s, a) = \mathbb{E}_{s' \sim P(s'|s,a)}[R_{t+1} + \gamma \max_{a'} Q^*(s', a') | S_t = s, A_t = a]$

## Dynamic Programming for RL
- **Assumption**: Requires a perfect model of the environment (i.e., known P and R).
- **Policy Iteration**: Iteratively improves a policy by alternating between:
    1.  **Policy Evaluation**: Compute $V^{\pi}(s)$ for the current policy $\pi$.
    2.  **Policy Improvement**: Update the policy $\pi$ greedily with respect to $V^{\pi}(s)$.
- **Value Iteration**: Directly computes the optimal value function $V^*(s)$ by iteratively applying the Bellman optimality equation until convergence.

## Model-Based vs. Model-Free RL
- **Model-Based RL**: The agent learns or is given a model of the environment (P and R) and uses it for planning.
- **Model-Free RL**: The agent learns directly from interactions with the environment without explicitly learning a model. This includes methods like Q-learning and SARSA.