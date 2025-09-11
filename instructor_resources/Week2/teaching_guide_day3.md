# Teaching Guide: Practical Applications - Day 3

## 🎯 Learning Objectives
By the end of this session, students should be able to:
1. Implement a complete data analysis pipeline
2. Perform feature engineering and selection
3. Build and evaluate machine learning models
4. Create an interactive dashboard
5. Present data insights effectively

## ⏰ Time Allocation (3-hour session)

### 1. Introduction (15 min)
- **Objective**: Understand the end-to-end data analysis process
- **Key Points**:
  - The data science workflow
  - Importance of each step in the pipeline
  - Real-world applications

### 2. Data Loading and Cleaning (30 min)

#### Concepts to Cover:
- Handling missing values
- Data type conversion
- Outlier detection and treatment
- Data validation

#### Teaching Approach:
1. **Live Demonstration** (15 min):
   ```python
   import pandas as pd
   import numpy as np
   
   # Load the Titanic dataset
   df = pd.read_csv('titanic.csv')
   
   # Handle missing values
   df['Age'].fillna(df['Age'].median(), inplace=True)
   df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
   
   # Feature engineering
   df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
   df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
   
   # Convert categorical variables
   df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
   ```

2. **Common Pitfalls**:
   - Data leakage
   - Information loss during imputation
   - Inconsistent data types

### 3. Exploratory Data Analysis (45 min)

#### Concepts to Cover:
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Feature importance

#### Teaching Approach:
1. **Interactive Exercise** (20 min):
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   # Set up the figure
   plt.figure(figsize=(15, 10))
   
   # Survival rate by passenger class
   plt.subplot(2, 2, 1)
   sns.barplot(x='Pclass', y='Survived', data=df)
   plt.title('Survival Rate by Passenger Class')
   
   # Age distribution by survival
   plt.subplot(2, 2, 2)
   sns.histplot(data=df, x='Age', hue='Survived', element='step', stat='density', common_norm=False)
   plt.title('Age Distribution by Survival')
   
   # Correlation heatmap
   plt.subplot(2, 2, 3)
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
   plt.title('Correlation Heatmap')
   
   # Fare vs Age colored by Survival
   plt.subplot(2, 2, 4)
   sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', alpha=0.6)
   plt.title('Fare vs Age by Survival')
   
   plt.tight_layout()
   plt.show()
   ```

2. **Discussion Points**:
   - Identifying patterns and relationships
   - Formulating hypotheses
   - Deciding on feature engineering steps

### 4. Feature Engineering and Selection (30 min)

#### Concepts to Cover:
- Creating new features
- Handling categorical variables
- Feature scaling
- Dimensionality reduction

#### Teaching Approach:
1. **Live Coding** (15 min):
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.feature_selection import SelectKBest, f_classif
   
   # Select features
   features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S']
   X = df[features]
   y = df['Survived']
   
   # Scale features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   # Feature selection
   selector = SelectKBest(score_func=f_classif, k=5)
   X_selected = selector.fit_transform(X_scaled, y)
   
   # Get selected features
   selected_features = [features[i] for i in selector.get_support(indices=True)]
   print(f"Selected features: {selected_features}")
   ```

### 5. Model Building and Evaluation (45 min)

#### Concepts to Cover:
- Train-test split
- Model selection
- Hyperparameter tuning
- Evaluation metrics

#### Teaching Approach:
1. **Live Demonstration** (20 min):
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
   
   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
   
   # Train the model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   
   # Make predictions
   y_pred = model.predict(X_test)
   
   # Evaluate the model
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))
   
   # Feature importance
   feature_importance = pd.DataFrame({
       'Feature': selected_features,
       'Importance': model.feature_importances_
   }).sort_values('Importance', ascending=False)
   
   plt.figure(figsize=(10, 6))
   sns.barplot(data=feature_importance, x='Importance', y='Feature')
   plt.title('Feature Importance')
   plt.show()
   ```

### 6. Creating an Interactive Dashboard (30 min)

#### Concepts to Cover:
- Introduction to Dash/Streamlit
- Layout design
- Adding interactive components
- Deploying the dashboard

#### Teaching Approach:
1. **Live Coding** (15 min):
   ```python
   import dash
   from dash import dcc, html
   import plotly.express as px
   
   app = dash.Dash(__name__)
   
   app.layout = html.Div([
       html.H1('Titanic Survival Analysis'),
       
       html.Div([
           html.Div([
               dcc.Graph(
                   id='survival-by-class',
                   figure=px.bar(df.groupby(['Pclass', 'Survived']).size().unstack(), 
                               title='Survival by Passenger Class')
               )
           ], className='six columns'),
           
           html.Div([
               dcc.Graph(
                   id='age-distribution',
                   figure=px.histogram(df, x='Age', color='Survived', 
                                     title='Age Distribution by Survival',
                                     barmode='overlay')
               )
           ], className='six columns')
       ], className='row'),
       
       dcc.Dropdown(
           id='feature-selector',
           options=[{'label': col, 'value': col} for col in ['Pclass', 'Sex_male', 'Embarked_Q', 'Embarked_S']],
           value='Pclass',
           style={'width': '50%'}
       ),
       
       dcc.Graph(id='feature-plot')
   ])
   
   @app.callback(
       dash.dependencies.Output('feature-plot', 'figure'),
       [dash.dependencies.Input('feature-selector', 'value')]
   )
   def update_plot(selected_feature):
       return px.box(df, x=selected_feature, y='Fare', color='Survived', 
                    title=f'Fare Distribution by {selected_feature} and Survival')
   
   if __name__ == '__main__':
       app.run_server(debug=True)
   ```

### 7. Presenting Insights (15 min)

#### Key Points:
- Storytelling with data
- Creating effective visualizations
- Communicating technical results to non-technical audiences
- Handling questions and feedback

### 8. Wrap-up and Q&A (15 min)

#### Discussion Questions:
1. What were the most important factors in predicting survival on the Titanic?
2. How could we improve our model's performance?
3. What other data would be useful to have?

## 🧑‍🏫 Teaching Tips

### Engagement Strategies:
- Use the Titanic dataset as a running example
- Encourage students to think critically about the data
- Have students work in pairs for some exercises

### Common Student Questions:
1. **Q**: How do I handle imbalanced classes?
   **A**: Consider techniques like SMOTE, class weights, or collecting more data for the minority class.

2. **Q**: When should I use a random forest vs. logistic regression?
   **A**: Use random forests for complex relationships and when you have many features, logistic regression for interpretability and when relationships are approximately linear.

3. **Q**: How many features should I select?
   **A**: It depends on your dataset size and the strength of the relationships. Use techniques like cross-validation to find the optimal number.

## 📝 Assessment Ideas

### In-class Activity:
- Have students work in groups to analyze a different aspect of the dataset
- Each group presents their findings to the class

### Homework:
- Apply the same analysis to a different dataset
- Write a report explaining the findings and recommendations

## 📚 Additional Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Dash Documentation](https://dash.plotly.com/)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
