"""Shared validated inference for Streamlit and the demonstration tests."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from run_analysis import FEATURES,NUM,CAT

BASE=Path(__file__).resolve().parent
PROVINCES=['WC','EC','NC','FS','KZN','NW','GT','MP','LP']
CATEGORIES=['A','B1','B2','B3','B4']

def validate_inputs(frame):
    missing=[c for c in FEATURES if c not in frame.columns]
    if missing:raise ValueError('Missing required columns: '+', '.join(missing))
    data=frame[FEATURES].copy()
    if len(data)==0:raise ValueError('Provide at least one municipality profile.')
    if len(data)>1000:raise ValueError('Upload at most 1,000 profiles at a time.')
    for c in NUM:
        data[c]=pd.to_numeric(data[c],errors='coerce')
        if not np.isfinite(data[c]).all():raise ValueError(c+': enter a finite numeric value for every row.')
        if c.endswith('_2011') or 'share' in c:
            if not data[c].between(0,100).all():raise ValueError(c+': percentages must be between 0 and 100.')
    if not data.province.isin(PROVINCES).all():raise ValueError('province: use WC, EC, NC, FS, KZN, NW, GT, MP or LP.')
    if not data.category.isin(CATEGORIES).all():raise ValueError('category: use A, B1, B2, B3 or B4.')
    if not data.household_size_2022.between(1,20).all():raise ValueError('household_size_2022: enter a value from 1 to 20.')
    if not data.log_households_2022.between(0,np.log1p(10_000_000)).all():raise ValueError('log_households_2022: outside the supported 0 to log(10,000,001) range.')
    if not data.household_growth_pct.between(-99.9,1000).all():raise ValueError('household_growth_pct: outside the supported range (-99.9 to 1,000%).')
    if (data[['formal_share_2022','traditional_share_2022','informal_share_2022']].sum(axis=1)>100.01).any():raise ValueError('Dwelling shares cannot add to more than 100%.')
    return data

def predict(frame):
    data=validate_inputs(frame);bundle=joblib.load(BASE/'final_model.joblib')
    probabilities=bundle['pipeline'].predict_proba(data)[:,1]
    return pd.DataFrame({'predicted_class':np.where(probabilities>=.5,'Higher access need','Lower access need'),
                         'higher_need_probability':probabilities,
                         'explanation':['Estimated 2022 category: access below 90% for water or sanitation.' if p>=.5 else 'Estimated 2022 category: access at least 90% for both services.' for p in probabilities]})
