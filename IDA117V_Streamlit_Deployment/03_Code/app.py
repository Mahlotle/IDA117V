"""Launch with: streamlit run app.py (from 03_Code)."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from predict import predict,PROVINCES,CATEGORIES

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Municipal access needs',layout='wide')
st.title('Municipal water and sanitation access needs')
st.write('Enter a municipal profile or upload profiles to estimate the approved 2022 access-need category.')
st.info('Research prototype based on 213 municipalities. Higher need means water or sanitation access below 90%. This is a project planning threshold, not an official Stats SA rating. Estimates are not future forecasts or household assessments. Use measured backlogs and human review for funding decisions.')
st.caption('Probabilities are model scores and have not been calibrated. Low or high confidence does not guarantee correctness.')
mode=st.radio('Input method',['Enter one profile','Upload CSV'],horizontal=True)
if mode=='Enter one profile':
    examples=pd.read_csv(ROOT/'02_Data/demo_valid_inputs.csv')
    choice=st.selectbox('Starting profile',['Example 1','Example 2','Example 3'])
    e=examples.iloc[int(choice[-1])-1]
    with st.form('profile'):
        left,right=st.columns(2)
        with left:
            province=st.selectbox('Province',PROVINCES,index=PROVINCES.index(e.province))
            category=st.selectbox('Municipal category',CATEGORIES,index=CATEGORIES.index(e.category),help='A: metro; B1: secondary city; B2: large town; B3: small town; B4: rural municipality.')
            households=st.number_input('Households in 2022',min_value=1,value=int(round(np.expm1(e.log_households_2022))),max_value=10_000_000)
            growth=st.number_input('Household growth 2011–2022 (%)',value=float(e.household_growth_pct))
            size=st.number_input('Average household size in 2022',value=float(e.household_size_2022))
            water=st.number_input('Improved water access in 2011 (%)',value=float(e.water_2011))
            sanitation=st.number_input('Improved sanitation access in 2011 (%)',value=float(e.sanitation_2011))
        with right:
            refuse=st.number_input('Refuse removal access in 2011 (%)',value=float(e.refuse_2011))
            electricity=st.number_input('Electricity proxy in 2011 (%)',value=float(e.electricity_2011),help='Census percentage using electricity for lighting.')
            formal=st.number_input('Formal dwellings in 2022 (%)',value=float(e.formal_share_2022))
            traditional=st.number_input('Traditional dwellings in 2022 (%)',value=float(e.traditional_share_2022))
            informal=st.number_input('Informal dwellings in 2022 (%)',value=float(e.informal_share_2022))
            urban=st.number_input('Urban households in 2022 (%)',value=float(e.urban_share_2022))
            rdp=st.number_input('RDP/subsidised dwelling share in 2022 (%)',value=float(e.rdp_share_2022),help='Share among the source table’s eligible dwelling denominator, not all households.')
        submitted=st.form_submit_button('Estimate access need')
    if submitted:
        row=dict(province=province,category=category,log_households_2022=np.log1p(households),household_growth_pct=growth,household_size_2022=size,water_2011=water,sanitation_2011=sanitation,refuse_2011=refuse,electricity_2011=electricity,formal_share_2022=formal,traditional_share_2022=traditional,informal_share_2022=informal,urban_share_2022=urban,rdp_share_2022=rdp)
        try:
            result=predict(pd.DataFrame([row]));st.success(result.predicted_class.iloc[0]);st.metric('Estimated probability of higher need',f'{result.higher_need_probability.iloc[0]:.1%}');st.write(result.explanation.iloc[0]);st.caption('The estimate combines historical service access with geography, housing and household context. It does not identify a causal reason for the service gap.')
        except ValueError as err:st.error(str(err))
else:
    template=(ROOT/'02_Data/demo_valid_inputs.csv').read_bytes()
    st.download_button('Download CSV input template',template,'municipal_profiles.csv','text/csv')
    upload=st.file_uploader('CSV with the same columns as the template',type=['csv'])
    if upload is not None:
        try:
            frame=pd.read_csv(upload);results=predict(frame);st.dataframe(results,use_container_width=True)
            st.download_button('Download predictions',results.to_csv(index=False),'predictions.csv','text/csv')
        except (ValueError,pd.errors.ParserError,pd.errors.EmptyDataError) as err:st.error(str(err))
st.caption('Source: Statistics South Africa, Report 03-01-83, Census 2011 and 2022. Independent student analysis. No personal records or credentials are required.')
