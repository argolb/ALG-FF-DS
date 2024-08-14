import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


bejaia = pd.read_csv('input/bejia-region-dataset.csv')
sba = pd.read_csv('input/sidi-bel-abbes-region-dataset.csv')

def combine_data(bejaia, sba):
    bejaia['region'] = 'bejaia'
    sba['region'] = 'sidi_bel_abbes'
    out = pd.concat([bejaia, sba], axis=0)
    out.columns = out.columns.str.lower().str.strip()
    return out


def clean_string_cols(df):
    alg = df.copy()
    for column in alg.columns:
        if alg[column].dtype == 'O':
            alg[column] = alg[column].str.strip()
    return alg

alg = combine_data(bejaia, sba)
# data cleaning:
alg = clean_string_cols(alg)
alg.classes.value_counts()
alg[['year', 'month', 'day']].drop_duplicates()

predictor_columns = [c for c in alg.columns if c != 'classes']
numeric_columns = ['day', 'month', 'year', 'temperature', 'rh', 'ws', 'rain', 'ffmc', 'dmc', 'isi', 'bui'] 

clean = alg[numeric_columns + ['classes']].dropna()
lr = LogisticRegression()
lr.fit(clean[numeric_columns], y=clean.classes)

clean['predictions'] = lr.predict(clean[numeric_columns])
clean[['classes', 'predictions']].value_counts().reset_index()


