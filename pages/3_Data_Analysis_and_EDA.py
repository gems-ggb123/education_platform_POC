import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.title('Exploratory Data Analysis and Preprocessing')

st.image('images/banner.png', use_column_width=True)

st.header('What is Exploratory Data Analysis (EDA)?')

st.markdown('''
Exploratory Data Analysis (EDA) is the process of analyzing and visualizing data to gain insights and understand its underlying structure. It involves summarizing the main characteristics of the data, identifying patterns, and detecting outliers or missing values.

EDA helps in understanding the distribution of variables, exploring relationships between variables, and identifying potential issues or biases in the data. It is an essential step in the data analysis process as it provides a foundation for further analysis and modeling.

''')

st.header('Why is EDA important?')

st.markdown('''
EDA is important for several reasons:

1. Data Understanding: EDA helps in understanding the data and its features. It provides insights into the range, distribution, and relationships between variables, which can guide further analysis and modeling decisions.

2. Data Cleaning: EDA helps in identifying missing values, outliers, or inconsistencies in the data. By detecting and addressing these issues, EDA ensures the data is clean and reliable for analysis.

3. Feature Selection: EDA helps in identifying relevant features or variables that are most informative for the analysis or modeling task. It helps in reducing dimensionality and improving model performance.

4. Hypothesis Generation: EDA can help in generating hypotheses or initial insights about the data. These hypotheses can be further tested and validated using statistical techniques or machine learning models.

''')

st.header('Preprocessing Steps')

st.markdown('''
Before performing any analysis or modeling, it is important to preprocess the data. Preprocessing involves transforming the raw data into a suitable format for analysis and modeling. Here are the preprocessing steps we will perform on the core data:
''')

st.markdown('''
1. It is important to drop any duplicated instances from the dataset.
''')

st.code('''
# Dropping Duplicates
core_data.drop_duplicates(inplace=True)
''')
# Importing the data
core_data = pd.read_csv('raw_data/core_data.csv')
# Only keep the cols, Site, Sect, Depth, NRG, Reflectance and Lithology
core_data = core_data.iloc[:, [1, 5, 6, 7, 8, 9, -1]]

# Dropping duplicates
n_duplicates = core_data.duplicated().sum()
core_data.drop_duplicates(inplace=True)
st.write(f'Dropped {n_duplicates} duplicates.')


st.markdown('''
2. Splitting into X and y: We will split the data into input features (X) and the target variable (y). The target variable will be the "Lithology" column.
''')
st.code('''
# Splitting into X and y
X = core_data.drop("Lithology", axis=1)
y = core_data["Lithology"]
''')

# Splitting into X and y
X = core_data.drop("Lithology", axis=1)
y = core_data["Lithology"]

st.table(X.head(5))

st.markdown('''
3. Train-Test Split: We will split the data into training and testing sets. The training set will be used to train the model, while the testing set will be used to evaluate its performance.
''')
st.code('''
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
''')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.table(X_train.head(5))

st.markdown('''
4. Dropping Columns: We will drop the "Site" and "Sect" columns as they are not relevant for our analysis.
''')
st.code('''
# Dropping columns
X_train.drop(["Site", "Sect"], axis=1, inplace=True)
X_test.drop(["Site", "Sect"], axis=1, inplace=True)
''')

# Dropping columns
X_train.drop(["Site", "Sect"], axis=1, inplace=True)
X_test.drop(["Site", "Sect"], axis=1, inplace=True)

st.table(X_train.head(5))


st.markdown('''
5. Creating a Correlation Plot: We will create a correlation plot using seaborn to visualize the correlation between the features.
''')
st.code('''
# Creating a correlation plot
fig = sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
st.pyplot(fig)
''')

# Creating a correlation plot
st.subheader('Correlation Plot')
with st.spinner('Plotting correlation plot...'):
    import matplotlib.pyplot as plt

    corr = X_train.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    # Add a label of the corr value:
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(i, j, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black')
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

st.markdown('''
6. Creating a Pairplot: We will create a pairplot using seaborn to visualize the relationships between the features.
''')
st.code('''
# Creating a pairplot
fig = sns.pairplot(X_train)
st.pyplot(fig)
''')


# Creating a pairplot
st.subheader('Pairplot of Features')
with st.spinner('Plotting pairplot...'):
    fig = sns.pairplot(X_train)
    st.pyplot(fig)



st.markdown('''
7. Creating a Pipeline: We will create a pipeline using scikit-learn to impute missing values and scale the features using a StandardScaler.
''')
st.code('''
# Creating a pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
''')

# Creating a pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Fit and transform the training data
X_train_preprocessed = pipeline.fit_transform(X_train)

st.write(pipeline)

st.markdown('''
These preprocessing steps are essential for preparing the data for analysis and modeling. By performing EDA and preprocessing, we can ensure the data is clean, relevant, and suitable for further analysis.

Now that we have completed the EDA and preprocessing steps, we can proceed with building machine learning models or performing any other analysis on the preprocessed data.

''')
