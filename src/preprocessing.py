"""
Modulo de Preprocesamiento y Preparacion de Datos para Deep Retention.
Maneja carga, limpieza, particion estratificada y transformaciones (OHE y Scaling)
sin fuga de informacion (data leakage).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_FILENAME = 'Customer-Churn-Records.csv'

NUMERICAL_FEATURES = (
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
)

CATEGORICAL_FEATURES = (
    'Geography',
    'Gender',
)

# Identificadores sin valor predictivo.
DROP_COLUMNS = ('RowNumber', 'CustomerId', 'Surname')

# 'Complain' coincide con 'Exited' en el 99.86% de los clientes: es un registro posterior
# (o simultaneo) al abandono. Usarla seria fuga de informacion (data leakage).
LEAKAGE_COLUMNS = ('Complain',)

# Correlacion con 'Exited' ~0 y la ablacion en Colab no mejora el ROC-AUC (0.8595 vs 0.8654).
NON_INFORMATIVE_COLUMNS = ('Satisfaction Score', 'Card Type', 'Point Earned')

TARGET_COLUMN = 'Exited'

EXCLUDED_COLUMNS = DROP_COLUMNS + LEAKAGE_COLUMNS + NON_INFORMATIVE_COLUMNS


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Carga el dataset bancario crudo desde archivo CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"El archivo no existe en la ruta: {data_path}")
    return pd.read_csv(data_path)


def assert_no_leakage(feature_columns) -> None:
    """Falla si alguna columna de fuga de informacion llega a las variables de entrada."""
    leaked = sorted(set(feature_columns) & set(LEAKAGE_COLUMNS + (TARGET_COLUMN,)))
    if leaked:
        raise ValueError(f"Columnas con fuga de informacion en las variables de entrada: {leaked}")


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las columnas que entran al modelo, en orden fijo."""
    features = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)
    assert_no_leakage(features)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas por el modelo: {missing}")
    return df[features].copy()


def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Particion estratificada en 3 conjuntos: Train (70%), Val (15%), Test (15%).
    Mantiene la proporcion de clientes que abandonan en cada conjunto.
    Devuelve las variables del modelo y conserva el indice original para poder
    recuperar los datos financieros de cada cliente.
    """
    X = get_model_features(df)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size,
        stratify=y_train_val, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_and_fit_preprocessor(X_train: pd.DataFrame, save_path: str = None) -> ColumnTransformer:
    """
    Construye y ajusta el preprocesador UNICAMENTE con los datos de entrenamiento.
    StandardScaler para variables numericas y OneHotEncoder para categoricas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(NUMERICAL_FEATURES)),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL_FEATURES)),
        ],
        remainder='drop'
    )
    preprocessor.fit(X_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
        print(f"Preprocesador guardado en: {save_path}")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Nombres legibles de todas las columnas de salida tras la transformacion."""
    cat_cols = list(preprocessor.named_transformers_['cat'].get_feature_names_out(list(CATEGORICAL_FEATURES)))
    return list(NUMERICAL_FEATURES) + cat_cols


def original_feature_name(encoded_name: str) -> str:
    """Mapea una columna transformada (ej. 'Geography_Germany') a su variable original."""
    for cat in CATEGORICAL_FEATURES:
        if encoded_name.startswith(cat + '_'):
            return cat
    return encoded_name


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transforma un conjunto de datos usando el preprocesador ya ajustado."""
    return preprocessor.transform(get_model_features(X)).astype(np.float32)


def preprocess_single_client(client_data: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """Prepara un unico registro de cliente para inferencia en tiempo real."""
    return transform_data(preprocessor, pd.DataFrame([client_data]))
