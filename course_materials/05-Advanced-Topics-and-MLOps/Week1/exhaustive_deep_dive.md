# Month 5, Week 1: Reinforcement Learning (Introduction) - Exhaustive Deep Dive

## Dynamic Programming

### Theoretical Explanation

Dynamic Programming (DP) is a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov Decision Process (MDP). The key idea of DP is the use of value functions to organize and structure the search for good policies.

**The Bellman Equation:**

The Bellman equation is a fundamental concept in reinforcement learning. It expresses the relationship between the value of a state and the values of its successor states. For a given policy π, the value of a state s is given by:

`Vπ(s) = E[Rt+1 + γVπ(St+1) | St=s]`

This equation can be decomposed into the immediate reward `Rt+1` and the discounted value of the successor state `γVπ(St+1)`.

**Policy Iteration:**

Policy iteration is an algorithm that finds the optimal policy by iteratively performing two steps:

1.  **Policy Evaluation:** For a given policy π, compute the state-value function Vπ.
2.  **Policy Improvement:** Improve the policy by acting greedily with respect to Vπ.

This process is guaranteed to converge to the optimal policy.

**Value Iteration:**

Value iteration is an algorithm that finds the optimal value function by iteratively updating the value function until it converges. The update rule for value iteration is given by:

`Vk+1(s) = max_a E[Rt+1 + γVk(St+1) | St=s, At=a]`

Once the optimal value function is found, the optimal policy can be extracted by acting greedily with respect to the optimal value function.

### Code Snippet: Value Iteration

```python
import numpy as np

# Example Grid World
# 0: start, 1: empty, 2: goal, 3: blocked
grid = np.array([
    [0, 1, 1, 3],
    [1, 1, 2, 3],
    [1, 1, 1, 1],
])

# Parameters
gamma = 0.9
theta = 1e-5

# Value function
V = np.zeros(grid.shape)

while True:
    delta = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 3 or grid[i, j] == 2:
                continue
            v = V[i, j]
            # Bellman update
            V[i, j] = max(
                # Up
                -1 + gamma * V[max(0, i - 1), j],
                # Down
                -1 + gamma * V[min(grid.shape[0] - 1, i + 1), j],
                # Left
                -1 + gamma * V[i, max(0, j - 1)],
                # Right
                -1 + gamma * V[i, min(grid.shape[1] - 1, j + 1)],
            )
            delta = max(delta, abs(v - V[i, j]))
    if delta < theta:
        break

print(V)
```

## Monte Carlo Methods

### Theoretical Explanation

Monte Carlo (MC) methods are a class of algorithms that learn from experience. They do not require a model of the environment and can be used to learn from complete episodes of interaction.

**First-Visit vs. Every-Visit MC:**

*   **First-Visit MC:** Estimates the value of a state by averaging the returns that are observed after the first visit to that state in each episode.
*   **Every-Visit MC:** Estimates the value of a state by averaging the returns that are observed after every visit to that state in each episode.

**Exploring Starts:**

To ensure that all state-action pairs are visited, MC methods with exploring starts begin each episode in a random state-action pair.

### Code Snippet: Monte Carlo Control

```python
import random

# Example Blackjack environment
# ...

# Q-table
Q = {}

# Returns
returns = {}

for _ in range(10000):
    # Generate an episode with exploring starts
    episode = []
    state = env.reset(exploring_starts=True)
    while True:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    # Update Q-table
    G = 0
    for state, action, reward in reversed(episode):
        G = gamma * G + reward
        if (state, action) not in [(s, a) for s, a, r in episode[:-1]]:
            if (state, action) not in returns:
                returns[(state, action)] = []
            returns[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns[(state, action)])
```

## Temporal-Difference Learning

### Theoretical Explanation

Temporal-Difference (TD) learning is a combination of MC methods and DP. Like MC methods, TD methods learn from experience. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for the final outcome (they bootstrap).

**TD(0) Update Rule:**

The TD(0) update rule is given by:

`V(St) = V(St) + α[Rt+1 + γV(St+1) - V(St)]`

where α is the learning rate.

**On-policy vs. Off-policy:**

*   **On-policy:** The agent learns about the policy that it is currently following.
*   **Off-policy:** The agent learns about a policy that is different from the policy that it is currently following.

**SARSA (On-policy):**

`Q(St, At) = Q(St, At) + α[Rt+1 + γQ(St+1, At+1) - Q(St, At)]`

**Q-learning (Off-policy):**

`Q(St, At) = Q(St, At) + α[Rt+1 + γmax_a Q(St+1, a) - Q(St, At)]`

### Code Snippet: Q-learning

```python
# Q-table
Q = np.zeros((state_space_size, action_space_size))

for _ in range(10000):
    state = env.reset()
    while True:
        # Epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _ = env.step(action)

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state
        if done:
            break
```

## Function Approximation

### Theoretical Explanation

Function approximation is a technique that is used to approximate the value function or the Q-function using a parameterized function, such as a neural network. This is useful for problems with large state spaces, where it is not feasible to store the value function or the Q-function in a table.

**Deep Q-Networks (DQNs):**

DQNs are a type of neural network that are used to approximate the Q-function. They are trained using a variant of Q-learning.

**DQN Algorithm:**

1.  Initialize the Q-network and the target network.
2.  For each episode:
    1.  Choose an action using an epsilon-greedy policy.
    2.  Take the action and observe the reward and the new state.
    3.  Store the experience in a replay buffer.
    4.  Sample a mini-batch of experiences from the replay buffer.
    5.  Compute the target Q-values using the target network.
    6.  Update the Q-network by minimizing the loss between the predicted Q-values and the target Q-values.
    7.  Periodically update the target network.

### Code Snippet: DQN with TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Q-network
model = Sequential([
    Dense(24, input_shape=(state_space_size,), activation='relu'),
    Dense(24, activation='relu'),
    Dense(action_space_size, activation='linear')
])

# Target network
target_model = Sequential([
    Dense(24, input_shape=(state_space_size,), activation='relu'),
    Dense(24, activation='relu'),
    Dense(action_space_size, activation='linear')
])

# ... (DQN training loop)
```