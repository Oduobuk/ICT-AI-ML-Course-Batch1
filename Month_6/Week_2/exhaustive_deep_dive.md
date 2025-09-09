# Month 6, Week 2: Exhaustive Deep Dive - Model-Free Reinforcement Learning

## Model-Free Reinforcement Learning - Learning from Experience

Model-free RL methods are crucial when the environment's dynamics (transition probabilities and reward function) are unknown or too complex to model explicitly. These methods learn directly from interactions with the environment, relying on observed rewards and state transitions.

## Monte Carlo (MC) Methods - Learning from Complete Episodes

MC methods are a class of model-free RL algorithms that learn value functions and optimal policies by averaging returns from complete episodes of experience. They are particularly useful when the environment is episodic (i.e., interactions naturally break into sequences that end).

### MC Prediction (Policy Evaluation):
To estimate the state-value function $V^{\pi}(s)$ or action-value function $Q^{\pi}(s, a)$ for a given policy $\pi$, MC methods average the returns observed after visits to state $s$ or state-action pair $(s, a)$.

- **First-Visit MC Prediction**: To estimate $V^{\pi}(s)$, the return is averaged only for the first time $s$ is visited in an episode. If $s$ is visited multiple times in an episode, only the return following the first visit is used.

- **Every-Visit MC Prediction**: To estimate $V^{\pi}(s)$, the return is averaged for every time $s$ is visited in an episode.

**Algorithm (First-Visit MC Prediction for $V^{\pi}$):
1.  Initialize $V(s)$ arbitrarily for all $s ∈ ℉$. (Note: \u2208 and \u2109 are unicode representations for 'element of' and 'script capital R' respectively, which are not standard markdown or text representations. They should be rendered as symbols if possible, or represented as their LaTeX equivalents if not.)
2.  Initialize `Returns(s)` as an empty list for all $s ∈ ℉$.
3.  For each episode:
    a.  Generate an episode following policy $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T, S_T$.
    b.  Initialize `G = 0`.
    c.  For $t = T-1$ down to $0$:
        i.   `G = R_{t+1} + \gamma G`
        ii.  If $S_t$ has not been visited before in this episode (from $S_0$ to $S_t$):
            1.  Append `G` to `Returns(S_t)`.
            2.  $V(S_t) = \text{average}(\text{Returns}(S_t))$

### MC Control (Policy Improvement):
MC control methods use MC prediction to evaluate a policy and then improve it, typically through generalized policy iteration (GPI).

**Algorithm (Monte Carlo ES - Exploring Starts):
1.  Initialize $Q(s, a)$ arbitrarily for all $s ∈ ℉, a ∈ ℂ$. (Note: \u2102 is unicode for 'script capital A', representing actions.)
2.  Initialize $\pi(s)$ arbitrarily for all $s ∈ ℉$.
3.  Repeat forever:
    a.  Generate an episode using exploring starts (random initial state-action pair) and following policy $\pi$.
    b.  For each pair $(s, a)$ appearing in the episode:
        i.   $G ←$ return following the first occurrence of $(s, a)$.
        ii.  Append $G$ to `Returns(s, a)`.
        iii. $Q(s, a) = \text{average}(\text{Returns}(s, a))$
    c.  For each state $s$ in the episode:
        i.   $̅(s) = ⁡\text{argmax}_a Q(s, a)$ (greedy policy improvement). (Note: \u0305 and \u2061 are combining characters and invisible characters respectively, which are not standard markdown or text representations. They should be rendered as symbols if possible, or represented as their LaTeX equivalents if not.)

## Temporal-Difference (TD) Learning - Bootstrapping from Experience

TD learning combines ideas from Monte Carlo and Dynamic Programming. Like MC, it learns directly from experience without a model. Like DP, it updates estimates based on other learned estimates (bootstrapping).

### TD(0) / One-step TD:
TD(0) updates value estimates after each step, based on the immediate reward and the estimated value of the next state. This is a key difference from MC, which waits until the end of an episode.

**Update Rule for $V(S_t)$:
$V(S_t) ← V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

Where:
- $\alpha$ is the learning rate.
- $R_{t+1} + \gamma V(S_{t+1})$ is the **TD target**.
- $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error**.

### Advantages of TD over MC:
- **Online Learning**: TD methods can learn after each step, making them suitable for continuous tasks or when immediate updates are needed.
- **Bootstrapping**: They use estimated values to update other estimated values, which can lead to faster learning and lower variance compared to MC.
- **Can learn from incomplete episodes**: Unlike MC, TD methods don't need to wait for the end of an episode.

## SARSA (State-Action-Reward-State-Action) - On-Policy TD Control

SARSA is an on-policy TD control algorithm that learns the action-value function $Q(s, a)$ for the policy being followed. The name SARSA comes from the tuple of events $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ that drives its update.

**Update Rule for $Q(S_t, A_t)$:
$Q(S_t, A_t) ← Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$

**Algorithm (SARSA):
1.  Initialize $Q(s, a)$ arbitrarily for all $s ∈ ℉, a ∈ ℂ$.
2.  Choose an initial state $S$.
3.  Choose an action $A$ from $S$ using an $ε$-greedy policy derived from $Q$. (Note: \u03B5 is unicode for 'epsilon'.)
4.  Repeat for each step of episode:
    a.  Take action $A$, observe reward $R$ and next state $S'$.
    b.  Choose action $A'$ from $S'$ using an $ε$-greedy policy derived from $Q$.
    c.  $Q(S, A) ← Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$
    d.  $S ← S'$, $A ← A'$.
5.  Until $S$ is terminal.

## Q-Learning - Off-Policy TD Control

Q-learning is an off-policy TD control algorithm that learns the optimal action-value function $Q^*(s, a)$ independently of the policy being followed. This means the agent can explore actions randomly while still learning the optimal policy.

**Update Rule for $Q(S_t, A_t)$:
$Q(S_t, A_t) ← Q(S_t, A_t) + \alpha [R_{t+1} + \gamma ⁡\max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$

**Algorithm (Q-Learning):
1.  Initialize $Q(s, a)$ arbitrarily for all $s ∈ ℉, a ∈ ℂ$.
2.  Choose an initial state $S$.
3.  Repeat for each step of episode:
    a.  Choose action $A$ from $S$ using an $ε$-greedy policy derived from $Q$.
    b.  Take action $A$, observe reward $R$ and next state $S'$.
    c.  $Q(S, A) ← Q(S, A) + \alpha [R + \gamma ⁡\max_{a'} Q(S', a') - Q(S, A)]$
    d.  $S ← S'$.
4.  Until $S$ is terminal.

### Key Difference: SARSA vs. Q-Learning
- **SARSA (On-Policy)**: The update for $Q(S_t, A_t)$ uses $Q(S_{t+1}, A_{t+1})$, where $A_{t+1}$ is the action *actually taken* by the current policy in state $S_{t+1}$. This means SARSA learns the value of the policy it is currently following, including its exploration strategy.
- **Q-Learning (Off-Policy)**: The update for $Q(S_t, A_t)$ uses $⁡\max_{a'} Q(S_{t+1}, a')$, which is the value of the *best possible action* in state $S_{t+1}$, regardless of what action the current policy would actually take. This allows Q-learning to learn the optimal policy even while exploring randomly.

## Exploration vs. Exploitation - The Fundamental Dilemma

- **Exploration**: The agent tries new actions or visits new states to gather more information about the environment and potentially discover better rewards.
- **Exploitation**: The agent uses its current knowledge to choose actions that are expected to yield the highest reward.

An agent must balance these two to maximize long-term reward. Too much exploration can lead to suboptimal performance, while too much exploitation can lead to getting stuck in local optima.

### Common Exploration Strategies:
- **$ε$-Greedy**: With probability $ε$, the agent chooses a random action (exploration). With probability $1-ε$, it chooses the greedy action (exploitation). $ε$ is often decayed over time.
- **Upper Confidence Bound (UCB)**: Selects actions based on an estimate of their value plus a term proportional to the uncertainty in that estimate. This encourages exploration of actions that have not been tried often or whose value estimates are highly uncertain.
- **Boltzmann Exploration (Softmax Exploration)**: Chooses actions probabilistically based on their estimated values, with higher values having a higher probability. A temperature parameter controls the level of exploration.

## Further Reading and References:
- **Sutton & Barto**: *"Reinforcement Learning: An Introduction"* (Chapter 6: Temporal-Difference Learning, Chapter 7: Multi-step Bootstrapping, Chapter 8: Planning and Learning with Tabular Methods).
- **Watkins & Dayan**: *"Q-Learning"* by Christopher J.C.H. Watkins and Peter Dayan (1992).
- **Rummery & Niranjan**: *"Online Q-Learning using Connectionist Networks"* by G.A. Rummery and M. Niranjan (1994) - introduces SARSA.

This exhaustive deep dive provides a comprehensive understanding of model-free reinforcement learning methods, including Monte Carlo, Temporal-Difference learning (SARSA and Q-learning), and the critical balance between exploration and exploitation.