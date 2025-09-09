# Month 6, Week 2: Deep Dive - Model-Free Reinforcement Learning

## Monte Carlo (MC) Methods
- **Concept**: Learn directly from episodes of experience. They estimate value functions and discover optimal policies by averaging returns from many simulated or actual episodes.
- **MC Prediction**: Estimating $V^{\pi}(s)$ or $Q^{\pi}(s, a)$ by averaging the returns observed after visits to state $s$ or state-action pair $(s, a)$.
- **First-Visit MC**: Averages returns only for the first time a state is visited in an episode.
- **Every-Visit MC**: Averages returns for every time a state is visited in an episode.
- **MC Control**: Uses MC prediction to evaluate a policy and then improves it, typically through generalized policy iteration (GPI).

## Temporal-Difference (TD) Learning
- **Concept**: Combines ideas from Monte Carlo and Dynamic Programming. Like MC, it learns directly from experience without a model. Like DP, it updates estimates based on other learned estimates (bootstrapping).
- **TD(0) / One-step TD**: Updates value estimates based on the immediate reward and the estimated value of the next state.
    $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$
- **Advantages over MC**: Can learn online (after each step), can learn from incomplete episodes, generally lower variance.

## SARSA (State-Action-Reward-State-Action)
- **On-Policy TD Control**: Learns the action-value function $Q(s, a)$ for the policy being followed.
- **Update Rule**: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$
- **Key**: The next action $A_{t+1}$ is chosen by the *same* policy that chose $A_t$.

## Q-Learning
- **Off-Policy TD Control**: Learns the optimal action-value function $Q^*(s, a)$ independently of the policy being followed.
- **Update Rule**: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$
- **Key**: The update uses the maximum possible Q-value for the next state, regardless of what action the current policy would actually take.

## On-Policy vs. Off-Policy Learning
- **On-Policy**: The policy being evaluated or improved is the same as the policy used to generate the data (e.g., SARSA).
- **Off-Policy**: The policy being evaluated or improved is different from the policy used to generate the data (e.g., Q-learning).

## Exploration vs. Exploitation
- **Exploration**: Trying new actions to discover more about the environment and potentially find better rewards.
- **Exploitation**: Using the current knowledge to choose actions that are expected to yield the highest reward.
- **Dilemma**: An agent must balance these two to maximize long-term reward.
- **Strategies**: Epsilon-greedy, Upper Confidence Bound (UCB), Boltzmann exploration.