# Week 1 Quiz: Introduction to AI/ML

## Instructions
- **Time Limit:** 30 minutes
- **Total Points:** 100
- **Attempts Allowed:** 2
- **Due:** End of Week 1

## Multiple Choice Questions (5 points each)

1. **Which of the following best describes the difference between AI and Machine Learning?**
   A) AI is a subset of ML  
   B) ML is a subset of AI  
   C) They are the same thing  
   D) They are completely unrelated  
   
   **Answer:** B

2. **What is the primary purpose of splitting data into training and test sets?**
   A) To increase the amount of training data  
   B) To evaluate how well the model generalizes to unseen data  
   C) To reduce the training time  
   D) To increase model complexity  
   
   **Answer:** B

3. **Which of the following is NOT a supervised learning task?**
   A) Classification  
   B) Regression  
   C) Clustering  
   D) None of the above  
   
   **Answer:** C

4. **What does the term 'overfitting' mean in machine learning?**
   A) The model is too simple to capture patterns in the data  
   B) The model performs well on training data but poorly on test data  
   C) The model performs equally well on training and test data  
   D) The model doesn't learn anything from the data  
   
   **Answer:** B

5. **Which Python library is primarily used for numerical operations on large arrays?**
   A) Pandas  
   B) NumPy  
   C) Matplotlib  
   D) Scikit-learn  
   
   **Answer:** B

## Short Answer Questions (15 points each)

6. **Explain the difference between supervised and unsupervised learning. Provide one example of each.**

   **Answer:**  
   Supervised learning uses labeled data to train models where the correct answers are provided. Example: Predicting house prices based on features like size and location.  
   Unsupervised learning finds patterns in unlabeled data. Example: Grouping customers into segments based on purchasing behavior.

7. **What is the purpose of feature scaling in machine learning? Name two common scaling techniques.**

   **Answer:**  
   Feature scaling standardizes the range of independent variables. Two techniques:  
   - Standardization (subtract mean, divide by standard deviation)  
   - Min-Max scaling (scale to a specific range, typically 0-1)

## Coding Exercise (30 points)

8. **Complete the following Python function that loads the Iris dataset, splits it into training and test sets, and returns them:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Load the Iris dataset and split into training and test sets.
    
    Args:
        test_size: Proportion of the dataset to include in the test split
        random_state: Controls the shuffling applied to the data
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Your code here
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
```

## Extra Credit (5 points)

9. **Explain one ethical consideration that should be taken into account when developing machine learning models.**

   **Answer:**  
   One ethical consideration is bias in the data or model. If the training data contains biases (e.g., underrepresentation of certain groups), the model may make unfair or discriminatory predictions. It's important to ensure diverse and representative training data and to regularly audit models for biased behavior.

## Submission Instructions
1. Save this file as `week1_quiz_<your_name>.md`
2. Fill in your answers directly in this document
3. Submit via the course learning management system by the due date
