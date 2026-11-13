import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

DATA_PATH = os.path.join(ROOT, 'data', 'Customer-Churn-Records.csv')


@pytest.fixture(scope='session')
def real_data():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def synthetic_data():
    """Dataset pequeno con el mismo esquema que el del profesor."""
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        'RowNumber': np.arange(n),
        'CustomerId': 1_000_000 + np.arange(n),
        'Surname': ['X'] * n,
        'CreditScore': rng.randint(350, 850, n),
        'Geography': rng.choice(['France', 'Germany', 'Spain'], n),
        'Gender': rng.choice(['Female', 'Male'], n),
        'Age': rng.randint(18, 90, n),
        'Tenure': rng.randint(0, 11, n),
        'Balance': rng.uniform(0, 200_000, n),
        'NumOfProducts': rng.randint(1, 5, n),
        'HasCrCard': rng.randint(0, 2, n),
        'IsActiveMember': rng.randint(0, 2, n),
        'EstimatedSalary': rng.uniform(10_000, 200_000, n),
    })
    logit = 0.08 * (df['Age'] - 40) - 1.0 * df['IsActiveMember'] - 1.2
    df['Exited'] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df['Complain'] = df['Exited']
    df['Satisfaction Score'] = rng.randint(1, 6, n)
    df['Card Type'] = rng.choice(['SILVER', 'GOLD'], n)
    df['Point Earned'] = rng.randint(100, 1000, n)
    return df
