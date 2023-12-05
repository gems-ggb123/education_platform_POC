import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title('Introduction to Logistic Regression')

st.image('images/banner.png', use_column_width=True)

st.header('What is Logistic Regression?')

st.markdown('''
Logistic Regression is a statistical model used to predict the probability of a binary outcome based on one or more predictor variables. It is commonly used for classification tasks where the target variable has two possible outcomes (e.g., yes/no, true/false).

Unlike linear regression, which predicts continuous values, logistic regression predicts the probability of an event occurring. The predicted probability is then converted into a binary outcome using a threshold value.

Logistic regression assumes a linear relationship between the predictor variables and the log-odds of the outcome. It uses a logistic function (sigmoid function) to map the linear combination of predictors to the probability of the outcome.

''')

st.header('Why is Logistic Regression important?')

st.markdown('''
Logistic Regression is widely used in various domains for binary classification tasks. Here are some reasons why it is important:

1. Interpretable: Logistic Regression provides interpretable results, allowing us to understand the impact of each predictor variable on the probability of the outcome.

2. Simple and Efficient: Logistic Regression is a relatively simple and computationally efficient algorithm. It can handle large datasets and is less prone to overfitting compared to more complex models.

3. Probability Estimation: Logistic Regression provides probability estimates, which can be useful for decision-making and risk assessment.

4. Feature Importance: Logistic Regression can help identify the most important features or variables that contribute to the prediction of the outcome.

''')

st.header('Model Training and Evaluation')

st.markdown('''
To train a logistic regression model, we need a labeled dataset with predictor variables and corresponding binary outcomes. The dataset is typically split into training and testing sets.

The logistic regression model is trained on the training set using an optimization algorithm (e.g., maximum likelihood estimation). Once trained, the model can be used to predict the outcomes for new, unseen data.

The performance of the logistic regression model can be evaluated using various metrics such as accuracy, precision, recall, and F1 score. These metrics provide insights into the model's predictive power and its ability to correctly classify the outcomes.

''')

st.header('Assumptions of Logistic Regression')

st.markdown('''
Logistic Regression makes several assumptions:

1. Linearity: Logistic Regression assumes a linear relationship between the predictor variables and the log-odds of the outcome. This assumption can be checked using techniques like scatter plots or residual analysis.

2. Independence: Logistic Regression assumes that the observations are independent of each other. This assumption can be violated in cases where the data has a hierarchical or clustered structure.

3. Absence of Multicollinearity: Logistic Regression assumes that the predictor variables are not highly correlated with each other. Multicollinearity can lead to unstable estimates and difficulties in interpreting the model.

4. Large Sample Size: Logistic Regression performs well with a large sample size. As the sample size increases, the estimates become more stable and the model's performance improves.

''')

st.header('Implementing the Model')

# Importing the data
core_data = pd.read_csv('raw_data/core_data.csv')
# Only keep the cols, Site, Sect, Depth, NRG, Reflectance and Lithology
core_data = core_data.iloc[:, [1, 5, 6, 7, 8, 9, -1]]

# Dropping duplicates
core_data.drop_duplicates(inplace=True)

# Drop any rows with missing values for Lithology
core_data.dropna(subset=['Lithology'], inplace=True)


# Splitting into X and y
X = core_data.drop("Lithology", axis=1)
y = core_data["Lithology"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Dropping columns
X_train.drop(["Site", "Sect"], axis=1, inplace=True)
X_test.drop(["Site", "Sect"], axis=1, inplace=True)



# Creating a pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
])


st.markdown('''
1. Training the Model: We will train the logistic regression model using the preprocessed training data.
''')
st.write(pipeline)
st.code('''
# Training the model
pipeline.fit(X_train, y_train)
''')

# Training the model
pipeline.fit(X_train, y_train)

st.markdown('''
2. Predicting on Training Data: We will use the trained model to make predictions on the training data.
''')
st.code('''
# Predicting on training data
y_train_pred = pipeline.predict(X_train)
y_train_pred[:10]
''')

# Predicting on training data
y_train_pred = pipeline.predict(X_train)
st.table(y_train_pred[:10])

st.markdown('''
3. Evaluating the Model: We will evaluate the performance of the model by calculating the accuracy score on the training data.
''')
st.code('''
# Calculating accuracy score
accuracy = accuracy_score(y_train, y_train_pred)
st.write("Accuracy on training data:", accuracy)
''')

# Calculating accuracy score
accuracy = accuracy_score(y_train, y_train_pred)
st.write("Accuracy on training data:", accuracy)

st.markdown('''
These preprocessing and modeling steps demonstrate the construction of a multiclass logistic regression model using scikit-learn. By performing EDA, preprocessing, and model training, we can build a predictive model that can classify the lithology based on the input features.

''')
