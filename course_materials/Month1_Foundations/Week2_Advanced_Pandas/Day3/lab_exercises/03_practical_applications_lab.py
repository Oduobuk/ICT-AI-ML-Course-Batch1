"""
Practical Applications Lab: End-to-End Data Analysis Pipeline
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Exercise 1: Data Loading and Initial Exploration
def load_and_explore_data():
    """
    Load and perform initial exploration of the dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Load the dataset
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    
    # Display basic information
    print("1. Dataset Overview:")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Display missing values
    print("\n2. Missing Values:")
    print("-" * 50)
    print(df.isnull().sum())
    
    # Display basic statistics
    print("\n3. Basic Statistics:")
    print("-" * 50)
    print(df.describe(include='all'))
    
    return df

# Exercise 2: Data Cleaning and Preprocessing
def clean_and_preprocess(df):
    """
    Clean and preprocess the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    # Make a copy of the original dataframe
    df_clean = df.copy()
    
    # 1. Handle missing values
    # Fill missing ages with median age by passenger class
    df_clean['Age'] = df_clean.groupby('Pclass')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fill missing embarked with mode
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Fill missing fare with median of passenger's class
    df_clean['Fare'] = df_clean.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # 2. Feature engineering
    # Extract title from name
    df_clean['Title'] = df_clean['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Dr': 'Rare',
        'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Lady': 'Rare',
        'Don': 'Rare', 'Dona': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Capt': 'Rare', 'Rev': 'Rare'
    }
    df_clean['Title'] = df_clean['Title'].replace(title_mapping)
    
    # Create family size feature
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    
    # Create is alone feature
    df_clean['IsAlone'] = 0
    df_clean.loc[df_clean['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Create age groups
    df_clean['AgeGroup'] = pd.cut(
        df_clean['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
    )
    
    # Create fare groups
    df_clean['FareGroup'] = pd.qcut(
        df_clean['Fare'],
        4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Drop unnecessary columns
    df_clean.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    return df_clean

# Exercise 3: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """
    Perform exploratory data analysis and create visualizations.
    
    Args:
        df: Input DataFrame
    """
    # Set up the figure
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Survival rate by passenger class
    plt.subplot(2, 2, 1)
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Survival Rate by Passenger Class')
    
    # Plot 2: Survival rate by gender
    plt.subplot(2, 2, 2)
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival Rate by Gender')
    
    # Plot 3: Age distribution by survival
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='Age', hue='Survived', element='step', stat='density', common_norm=False)
    plt.title('Age Distribution by Survival')
    
    # Plot 4: Heatmap of correlations
    plt.subplot(2, 2, 4)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # Interactive visualization with Plotly
    fig = px.sunburst(
        df,
        path=['Pclass', 'Sex', 'AgeGroup', 'Survived'],
        values='Fare',
        title='Interactive Sunburst Chart: Survival by Class, Gender, and Age Group'
    )
    fig.write_html('survival_sunburst.html')
    print("\nInteractive sunburst chart saved as 'survival_sunburst.html'")

# Exercise 4: Build a Predictive Model
def build_model(df):
    """
    Build and evaluate a predictive model.
    
    Args:
        df: Input DataFrame
    """
    # Select features and target
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']]
    y = df['Survived']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define preprocessing for numeric and categorical features
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print("-" * 50)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance
    feature_importances = model.named_steps['classifier'].feature_importances_
    
    # Get feature names after one-hot encoding
    ohe_columns = list(model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features))
    
    all_features = numeric_features + ohe_columns
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.show()
    
    return model

def main():
    # Exercise 1: Load and explore data
    print("1. Loading and exploring data...")
    df = load_and_explore_data()
    
    # Exercise 2: Clean and preprocess data
    print("\n2. Cleaning and preprocessing data...")
    df_clean = clean_and_preprocess(df)
    
    # Display cleaned data info
    print("\nCleaned Data Info:")
    print("-" * 50)
    print(df_clean.info())
    
    # Exercise 3: Perform EDA
    print("\n3. Performing exploratory data analysis...")
    perform_eda(df_clean)
    
    # Exercise 4: Build and evaluate model
    print("\n4. Building and evaluating predictive model...")
    model = build_model(df_clean)
    
    print("\nLab exercise completed!")

if __name__ == "__main__":
    main()
