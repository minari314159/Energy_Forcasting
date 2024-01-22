import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
predictions = st.container()
sidebar = st.container()

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with sidebar:
    st.sidebar.header('Energy Consumption `Forcasting` ')

    st.sidebar.caption('PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system. This includes the following states Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.')
    st.sidebar.caption("This analysis and predictions with focus on PJME and PJMW as they are the most comprehensive data set, although the general regions are larger thus the results will be more generalized")

    st.sidebar.divider()

    st.sidebar.markdown('''
    ---
    ''')
    st.sidebar.link_button("<- back to portfolio",
                        url="https://3-d-portfolio-pi.vercel.app", type='secondary')
    st.sidebar.markdown('| Created SJ Olsen ')
    st.sidebar.markdown('`Data published online at OurWorldInData.org`')

with header:
    st.title("Forcasting of Energy Consumption Data")

with dataset:
    st.header('PJM Hourly Energy Consumption Data')
    df_East = pd.read_csv('DataSet/Regions/PJME_hourly.csv').set_index('Datetime')
    df_East.index = pd.to_datetime(df_East.index)
    st.write(df_East.describe())

with features:
    st.header('Features')

    st.markdown('* **First Feature**: Why this feature, how was it calculated')
    st.markdown('* **Second Feature**: Why this feature, how was it calculated')
    st.markdown('* **Third Feature**: Why this feature, how was it calculated')

with modelTraining:
    st.header('Model Training')

with predictions:
    st.header('Predictions')
