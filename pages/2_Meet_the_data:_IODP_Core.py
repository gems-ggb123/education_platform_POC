
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


st.title('Introduction to Geological Core Data')

st.image('images/banner.png', use_column_width=True)

st.image('images/geological_core.jpg', caption='Image representing geological core data. Source: Texas A&amp;M (no date) Core Photographs and Digital Images, IODP JRSO  Core Photographs and Digital Images. Available at: https://iodp.tamu.edu/database/coreimages.html (Accessed: 05 December 2023). ', use_column_width=True)

st.header('About the Data')

st.markdown('''
The data used in this project is from the International Ocean Discovery Program (IODP) and contains geological core data. The dataset includes features such as depth, natural gamma, reflectance, and a target column of Lithology.
''')

core_data = pd.read_csv('raw_data/core_data.csv')
# Only keep the cols, Site, Sect, Depth, NRG, Reflectance and Lithology
core_data = core_data.iloc[:, [1, 5, 6, 7, 8, 9, -1]]

st.dataframe(core_data.head())

st.markdown('''Geological core data plays a crucial role in understanding the Earth's subsurface and its geological history. It provides valuable insights into the composition, structure, and properties of rocks and sediments.

By analyzing core data, geologists can identify different lithologies, understand depositional environments, and make interpretations about past climate conditions and geological processes.

In this project, we will explore the application of machine learning techniques to analyze and interpret geological core data. By leveraging the power of machine learning, we can uncover hidden patterns, make predictions, and gain valuable insights from this rich dataset.

Let's dive into the fascinating world of geological core data and its applications in geoscience research and exploration!
''')

core_data = pd.read_csv('raw_data/core_data.csv')
st.dataframe(core_data)
