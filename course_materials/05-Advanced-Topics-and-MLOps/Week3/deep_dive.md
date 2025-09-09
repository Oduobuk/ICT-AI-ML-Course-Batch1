
# Month 5, Week 3: Explainable and Responsible AI (XAI) - Deep Dive

## Addressing Bias in Data and Models

*   **Types of Bias:**
    *   **Selection Bias:** Occurs when the data is not representative of the population.
    *   **Measurement Bias:** Occurs when the data is collected in a way that is not accurate or consistent.
    *   **Algorithmic Bias:** Occurs when the algorithm itself is biased.

*   **Mitigation Strategies:**
    *   **Data Augmentation:** A technique for increasing the size of the dataset by creating new data points from the existing data.
    *   **Fairness Constraints:** A set of constraints that are added to the optimization problem to ensure that the model is fair.
    *   **Adversarial Debiasing:** A technique that uses an adversary to learn a representation of the data that is invariant to sensitive attributes.

## Interpretability vs. Explainability

*   **Interpretability:** The ability to understand the inner workings of a model.
*   **Explainability:** The ability to explain the predictions of a model in a way that is understandable to humans.

## Techniques like LIME and SHAP

*   **LIME (Local Interpretable Model-agnostic Explanations):** A technique that explains the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable model locally around the prediction.
*   **SHAP (SHapley Additive exPlanations):** A game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.
