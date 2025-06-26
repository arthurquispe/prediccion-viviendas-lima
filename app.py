
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------ Cargar dataset ------------------
df = pd.read_csv("Precio_viviendas_lima_peru_by_AQC.csv")
cat_cols = ['Tipo de Propiedad', 'Estado', 'Zona_macro']
num_cols = ['Dormitorios', 'Baños', 'Garajes', 'M² edificados']

# ------------------ Transformaciones target ------------------
from scipy.stats.mstats import winsorize
df['Precio_filtrado'] = winsorize(df['Precio'], limits=[0.01, 0.01])
df['Precio_sqrt'] = np.sqrt(df['Precio_filtrado'])
scaler_y = RobustScaler()
y_scaled = scaler_y.fit_transform(df['Precio_sqrt'].values.reshape(-1, 1)).ravel()

X = df.drop(columns=['Unnamed: 0', 'Precio', 'Precio_filtrado', 'Precio_sqrt'])

# ------------------ Preprocesamiento ------------------
numeric_transformer = Pipeline([('scaler', RobustScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
preprocessor.fit(X)

# ------------------ PyTorch model ------------------
class PriceModel(nn.Module):
    def __init__(self, input_dim):
        super(PriceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

# Cargar modelo PyTorch entrenado (puedes ajustar el path)
model_pytorch = PriceModel(preprocessor.transform(X[:1]).shape[1])
model_pytorch.load_state_dict(torch.load("modelo_pytorch.pth", map_location=torch.device('cpu')))
model_pytorch.eval()

# Cargar modelo sklearn (ajustar si tienes un archivo .pkl guardado)
model_sklearn = joblib.load("modelo_sklearn.pkl")

# ------------------ Streamlit UI ------------------
st.title("Predicción del Precio de Viviendas en Lima")

modelo_tipo = st.selectbox("Selecciona el modelo", ["Scikit-learn (MLPRegressor)", "PyTorch"])

# Inputs del usuario
input_data = {}
for col in num_cols:
    input_data[col] = st.number_input(col, value=1 if col != "M² edificados" else 50)

input_data['Tipo de Propiedad'] = st.selectbox("Tipo de Propiedad", df['Tipo de Propiedad'].unique())
input_data['Estado'] = st.selectbox("Estado", df['Estado'].unique())
input_data['Zona_macro'] = st.selectbox("Zona macro", df['Zona_macro'].unique())

# Cuando se presiona el botón
if st.button("Predecir Precio"):
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)

    if modelo_tipo.startswith("Scikit"):
        pred_scaled = model_sklearn.predict(input_df)  # <- sin transformar antes
    else:
        with torch.no_grad():
            input_tensor = torch.tensor(input_transformed, dtype=torch.float32)
            pred_scaled = model_pytorch(input_tensor).numpy().ravel()

    pred_precio = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel() ** 2
    st.success(f"Precio estimado: USD $ {pred_precio[0]:,.2f}")
