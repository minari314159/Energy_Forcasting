
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go 
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from DataSet.load_data import load_PJME, load_PJMW
from create_model import create_model


#page outline
header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
future = st.container()
sidebar = st.container()

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with sidebar:
    st.sidebar.header('Energy Consumption `Forcasting` ')

    st.sidebar.caption('PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system. This includes the following states Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.')
    st.sidebar.caption("This analysis and predictions with focus on PJME and PJMW as they are the most comprehensive data set, although the general regions are larger thus the results will be more generalized")

    st.sidebar.divider()
    west_east = st.sidebar.selectbox(
        'Choose west or east dataset', options=['PJM West Region', 'PJM East Region'], index=1)
    n_estimators = st.sidebar.slider(
        'Choose n estimators for the model', min_value=500, max_value=1000, value=500)
    future_predictions = st.sidebar.selectbox('How far in the future to predict', options=[0.5,1,2], index=1)

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
     
    col1, col2 = st.columns((6,4))
    df_East = load_PJME()
    df_West = load_PJMW()

    if west_east == 'PJM East Region':
        data = df_East
    else:
        data = df_West

    with col1:
        fig = px.line(data, x=data.index,
                    y=data['Energy Use (MW)'])

        fig.update_layout(
            showlegend= False,
            width=800,
            height=500,
            title='Energy Consumption 2002-2018',
            xaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            )
        )
        fig.update_xaxes(minor=dict(ticks="outside", showgrid=True),
                        rangeslider_visible=True)
        fig.update_traces(line_color="#f7768e")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            data, x=data['Energy Use (MW)'], nbins=500, 
            width=400,
            height=350,
            title='Frequency of Energy Consumption',
            color_discrete_sequence=['#f7768e'])
        st.plotly_chart(fig, use_container_width=True)
        

with features:
    st.header('Features')
    col1,col2 = st.columns((3,7))
    with col1:
        st.markdown('* **Hour Feature**')
        st.markdown('* **Month Feature**')
        st.markdown('* **Year Feature**')
    
    with col2:
        reg = create_model(data, n_estimators)[0]

        fi = pd.DataFrame(data=reg.feature_importances_,
                          index=reg.feature_names_in_,
                          columns=['Importance'])
        
        fig = px.bar(fi.iloc[0:3], labels={'index':'Feature', 'value':'Importance'})
        fig.update_layout(
            showlegend=False,
            width=700,
            height=300,
            title='Feature Importance',
            xaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            )
        )
        fig.update_traces(marker_color='#f7768e')
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown('**Lag Features** were calculated for 1,2 and 3 years. Indicating the validity of predicted data from 1-3 years in advance')
    fig = px.bar(fi.iloc[3:], labels={'index':'Lag Features', 'value':'Importance'})
    fig.update_layout(
            showlegend=False,
            width=700,
            height=300,
            title='Lag Importance',
            xaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            )
        )
    fig.update_traces(marker_color='#f7768e')
    st.plotly_chart(fig, use_container_width=True)

with modelTraining:
    st.header('Model Training')
    
    df_preds = pd.DataFrame(create_model(data, n_estimators)[1])
    df_preds['prediction'] = df_preds['prediction'].fillna(0)
    df_preds = df_preds.query('prediction > 19_000')
  
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_preds.index,
                  y=df_preds['prediction'], 
                  name='Predicted Set', 
                  mode='lines',
                  line={'color': '#f7768e'}))

    fig.add_trace(go.Scatter(x=df_preds.index,
                    y=df_preds['Energy Use (MW)'],
                    name='Test Set',
                    mode='lines',
                    line={'color': '#8576f7'},
                    opacity=0.5))
    fig.update_layout(
            showlegend= True,
            width=800,
            height=500,
            title='Energy Consumption 20-2018',
            xaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='#787a79',
                linewidth=1
            )
        )
    fig.update_xaxes(minor=dict(ticks="outside", showgrid=True),
                        rangeslider_visible=True)
    
    st.plotly_chart(fig, use_container_width=True)

    score = mean_absolute_percentage_error(
        df_preds['Energy Use (MW)'], 
        df_preds['prediction'])
    st.write(f'MAPE Score:` {score:0.2f}`')

    st.markdown('### Improvements')
    st.markdown('**The following are suggested improvements to lower the RMSE score:**')
    st.markdown("* Add features including holidays, moderate to extreme weather ")
    st.markdown("* Both of these will improve the predictions for those 'outlier' days during the month of varying energy consumption that don't follow the seasonal pattern")
    st.markdown("* Reduce over fitting inherent to XGBoost models")

with future:
    st.header('Forcasting')
    
