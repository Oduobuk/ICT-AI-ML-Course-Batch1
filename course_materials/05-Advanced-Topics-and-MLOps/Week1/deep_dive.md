
# Month 5, Week 1: Reinforcement Learning (Introduction) - Deep Dive

## Markov Decision Processes (MDPs)

*   **Components of an MDP:**
    *   **States (S):** A set of states that the agent can be in.
    *   **Actions (A):** A set of actions that the agent can take.
    *   **Transition Probabilities (P):** The probability of transitioning from one state to another after taking a certain action.
    *   **Rewards (R):** The reward that the agent receives for transitioning from one state to another.
    *   **Discount Factor (γ):** A value between 0 and 1 that determines the importance of future rewards.

*   **The Bellman Equation:** A fundamental equation in reinforcement learning that relates the value of a state to the value of its successor states.

## Q-learning and Value Functions

*   **Value Function:** A function that estimates the expected future reward that an agent will receive from being in a particular state.
*   **Q-function:** A function that estimates the expected future reward that an agent will receive from taking a particular action in a particular state.
*   **Q-learning Algorithm:**
    1.  Initialize the Q-table with zeros.
    2.  For each episode:
        1.  Choose an action.
        2.  Take the action and observe the reward and the new state.
        3.  Update the Q-table using the Bellman equation.

## Policy Gradients

*   **Policy:** A function that maps a state to an action.
*   **Policy Gradient Methods:** A class of reinforcement learning algorithms that learn a policy directly, without learning a value function. They work by adjusting the parameters of the policy in the direction that increases the expected future reward.
