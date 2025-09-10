"""
Netflix Movies and TV Shows Analysis - Solution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn')
sns.set_palette('husl')

def load_and_explore():
    """Load and explore the Netflix dataset."""
    # Load the dataset
    df = pd.read_csv('netflix_titles.csv')
    
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

def clean_data(df):
    """Clean and preprocess the dataset."""
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['director'].fillna('Unknown', inplace=True)
    df_clean['cast'].fillna('Unknown', inplace=True)
    df_clean['country'].fillna('Unknown', inplace=True)
    df_clean['date_added'] = pd.to_datetime(df_clean['date_added'].str.strip(), errors='coerce')
    
    # Extract year from date_added
    df_clean['year_added'] = df_clean['date_added'].dt.year
    
    # Create duration_minutes column
    df_clean['duration_minutes'] = df_clean.apply(
        lambda x: int(x['duration'].split()[0]) if 'min' in x['duration'] else None, axis=1
    )
    
    # Create duration_seasons column
    df_clean['duration_seasons'] = df_clean.apply(
        lambda x: int(x['duration'].split()[0]) if 'Season' in x['duration'] else None, axis=1
    )
    
    # Extract main genre
    df_clean['main_genre'] = df_clean['listed_in'].str.split(',').str[0].str.strip()
    
    return df_clean

def analyze_content_distribution(df):
    """Analyze the distribution of content on Netflix."""
    # Content type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='type', data=df)
    plt.title('Distribution of Content Types on Netflix')
    plt.xlabel('Content Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('content_type_distribution.png')
    plt.close()
    
    # Content by country (top 10)
    top_countries = df['country'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title('Top 10 Countries by Content Production')
    plt.xlabel('Number of Titles')
    plt.tight_layout()
    plt.savefig('content_by_country.png')
    plt.close()
    
    # Content added over time
    content_over_time = df.groupby(['year_added', 'type']).size().unstack()
    content_over_time.plot(kind='line', figsize=(12, 6))
    plt.title('Content Added to Netflix Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Titles')
    plt.legend(title='Content Type')
    plt.tight_layout()
    plt.savefig('content_over_time.png')
    plt.close()

def create_interactive_visualizations(df):
    """Create interactive visualizations using Plotly."""
    # Interactive scatter plot: Rating vs Release Year by Type
    fig1 = px.scatter(
        df,
        x='release_year',
        y='rating',
        color='type',
        hover_data=['title', 'country', 'duration'],
        title='Content Rating by Release Year and Type',
        labels={'release_year': 'Release Year', 'rating': 'Rating'}
    )
    fig1.write_html('rating_vs_year.html')
    
    # Interactive sunburst chart: Content by Country and Type
    country_type = df.groupby(['country', 'type']).size().reset_index(name='count')
    fig2 = px.sunburst(
        country_type,
        path=['country', 'type'],
        values='count',
        title='Content Distribution by Country and Type'
    )
    fig2.write_html('content_by_country_type.html')

def main():
    # 1. Load and explore the data
    print("Loading and exploring the dataset...")
    df = load_and_explore()
    
    # 2. Clean and preprocess the data
    print("\nCleaning and preprocessing the data...")
    df_clean = clean_data(df)
    
    # 3. Analyze content distribution
    print("\nAnalyzing content distribution...")
    analyze_content_distribution(df_clean)
    
    # 4. Create interactive visualizations
    print("\nCreating interactive visualizations...")
    create_interactive_visualizations(df_clean)
    
    print("\nAnalysis complete! Check the generated visualizations.")

if __name__ == "__main__":
    main()
