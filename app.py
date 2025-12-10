import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

# ============================================
# 1. CHARGER LE MODELE ET LES TRANSFORMERS
# ============================================

best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer_num_saved = joblib.load('imputer_num.pkl')
imputer_cat_saved = joblib.load('imputer_cat.pkl')
label_encoders = joblib.load('label_encoders.pkl')
x_columns = joblib.load('x_columns.pkl')
numeric_features = joblib.load('numeric_features.pkl')
categorical_features = joblib.load('categorical_features.pkl')

# --- RECREATION DU SIMPLE IMPUTER NUMERIQUE ---
# On recrée un SimpleImputer et on injecte les statistiques
imputer_num = SimpleImputer(strategy='median')
# Créer un DataFrame minimal pour refitter
dummy_df = pd.DataFrame([imputer_num_saved.statistics_], columns=numeric_features)
imputer_num.fit(dummy_df)

# Idem pour le SimpleImputer catégoriel
imputer_cat = SimpleImputer(strategy='most_frequent')
dummy_cat_df = pd.DataFrame([imputer_cat_saved.statistics_], columns=categorical_features)
imputer_cat.fit(dummy_cat_df)

st.set_page_config(page_title="Prédiction Valeur Joueur", layout="wide")
st.title("Prédiction de la Valeur d'un Joueur de Football")

# ============================================
# 2. CRÉER LES INPUTS UTILISATEUR
# ============================================

st.sidebar.header("Saisir les caractéristiques du joueur")
user_input = {}

# Inputs numériques
for i, col in enumerate(numeric_features):
    default_val = float(imputer_num.statistics_[i]) if hasattr(imputer_num, 'statistics_') else 0
    user_input[col] = st.sidebar.number_input(f"{col}", min_value=0.0, max_value=1e5, value=default_val)

# Inputs catégoriels
for col in categorical_features:
    le = label_encoders[col]
    categories = list(le.classes_)
    selected = st.sidebar.selectbox(f"{col}", categories)
    user_input[col] = selected

# Convertir en DataFrame
input_df = pd.DataFrame([user_input], columns=x_columns)

# ============================================
# 3. TRAITEMENT DES INPUTS
# ============================================

# Imputation numérique
input_df[numeric_features] = imputer_num.transform(input_df[numeric_features])

# Imputation catégorielle
input_df[categorical_features] = imputer_cat.transform(input_df[categorical_features])

# Encodage catégoriel
for col in categorical_features:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col].astype(str))

# Normalisation
input_scaled = scaler.transform(input_df)

# ============================================
# 4. PRÉDICTION
# ============================================

if st.button("Prédire la valeur du joueur"):
    prediction = best_model.predict(input_scaled)[0]
    st.success(f"La valeur estimée du joueur est : {prediction:,.0f} €")

    with st.expander("Détails de l'entrée utilisateur"):
        st.json(user_input)
