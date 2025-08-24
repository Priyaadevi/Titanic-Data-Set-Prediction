# Titanic-Data-Set-Prediction
Using Machine Learning 

#Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 1)Loading the dataset
df = pd.read_csv("/content/titanic.csv")

# Displaying the dataset
df

# 2)Cleaning the dataset
# Handle missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Handle missing Cabin by replacing with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

# Handle missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical columns into numeric codes
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 3) Visualizations
# Survival rate by gender
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.xticks([0, 1], ['Male', 'Female'])
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age distribution of survivors vs non-survivors
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution: Survivors vs Non-Survivors')
plt.show()

# Heatmap of correlations - only for numeric columns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

