
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('GeoLearn - POC Developed by James Brook 2023')

st.image('images/banner.png', use_column_width=True)

pages = ['Home', 'Introduction', 'Data Exploration']

st.header('Welcome to GeoLearn!')

st.markdown('''
GeoLearn is an educational web application that aims to teach geologists about machine learning and its applications in geology.

In this platform, you will learn how to use machine learning algorithms to analyze geological data, make predictions, and gain insights.

In this first version, we will focus on the application of machine learning in the field of core analysis. 

- Introduction to Machine Learning
- Data Exploration and Preprocessing
- Supervised Learning Algorithms
- Model Evaluation and Selection

With more content to be added in the future!
Let's get started and explore the fascinating world of machine learning for geologists!
''')
