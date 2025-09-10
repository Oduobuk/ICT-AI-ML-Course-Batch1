# Practical Applications of Pandas and Data Visualization

## Table of Contents
1. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Feature Engineering](#feature-engineering)
4. [Building a Data Pipeline](#building-a-data-pipeline)
5. [Creating a Dashboard](#creating-a-dashboard)

## Data Cleaning and Preparation

### Handling Missing Data
```python
# Check for missing values
df.isnull().sum()

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop columns with too many missing values
df.drop(columns=['cabin'], inplace=True)
```

### Data Type Conversion
```python
# Convert to categorical
df['survived'] = df['survived'].astype('category')
df['pclass'] = df['pclass'].astype('category')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
```

### Removing Outliers
```python
# Using IQR method
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['fare'] >= lower_bound) & (df['fare'] <= upper_bound)]
```

## Exploratory Data Analysis (EDA)

### Univariate Analysis
```python
# Numerical variables
df.describe()

# Categorical variables
df['survived'].value_counts(normalize=True) * 100
```

### Bivariate Analysis
```python
# Cross-tabulation
pd.crosstab(df['pclass'], df['survived'], normalize='index') * 100

# Correlation matrix
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

### Time Series Analysis
```python
# Resample by month
monthly_sales = df.resample('M', on='date')['amount'].sum()

# Rolling average
rolling_avg = df['amount'].rolling(window=7).mean()
```

## Feature Engineering

### Creating New Features
```python
# Age groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 18, 35, 60, 100],
    labels=['child', 'young_adult', 'adult', 'senior']
)

# Title extraction
df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```

### Encoding Categorical Variables
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['embarked', 'sex'])

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['embarked_encoded'] = le.fit_transform(df['embarked'])
```

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Min-Max scaling
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
```

## Building a Data Pipeline

### Using pandas' pipe
```python
def clean_data(df):
    # Data cleaning steps
    df = df.drop_duplicates()
    df = df.dropna(subset=['important_column'])
    return df

def feature_engineering(df):
    # Feature engineering steps
    df['new_feature'] = df['col1'] / df['col2']
    return df

# Apply pipeline
df_processed = (df
               .pipe(clean_data)
               .pipe(feature_engineering))
```

### Using scikit-learn's Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['age', 'fare']),
        ('cat', categorical_transformer, ['embarked', 'sex'])
    ])

# Create and fit the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(df)
```

## Creating a Dashboard

### Using Plotly Dash
```python
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create the layout
app.layout = html.Div([
    html.H1('Titanic Dashboard'),
    
    dcc.Dropdown(
        id='class-dropdown',
        options=[{'label': c, 'value': c} for c in df['class'].unique()],
        value='First',
        multi=True
    ),
    
    dcc.Graph(id='survival-plot')
])

# Define callbacks
@app.callback(
    Output('survival-plot', 'figure'),
    [Input('class-dropdown', 'value')]
)
def update_plot(selected_classes):
    filtered_df = df[df['class'].isin(selected_classes)]
    fig = px.histogram(
        filtered_df,
        x='age',
        color='survived',
        barmode='group',
        title='Age Distribution by Survival Status'
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```

## Best Practices

### Code Organization
1. **Modular Code**: Break code into functions and modules
2. **Configuration**: Use config files for parameters
3. **Logging**: Implement proper logging
4. **Documentation**: Add docstrings and comments

### Performance Optimization
1. Use vectorized operations
2. Avoid loops when possible
3. Use appropriate data types
4. Consider using Dask for large datasets

### Reproducibility
1. Set random seeds
2. Use version control
3. Document package versions
4. Use virtual environments

## Exercises
1. Clean and preprocess a messy dataset
2. Perform comprehensive EDA
3. Create new features from existing data
4. Build a data processing pipeline
5. Create an interactive dashboard

## Resources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Dash Documentation](https://dash.plotly.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
