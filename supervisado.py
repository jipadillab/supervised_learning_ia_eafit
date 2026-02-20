import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="DataScience Studio", layout="wide")

# --- TITULO Y SELECCIÓN DE DATASET ---
st.title("🔬 Laboratorio de Clasificación Avanzado")

dataset_name = st.sidebar.selectbox("1. Selecciona el Dataset", ("Iris", "Wine", "Breast Cancer"))

def load_data(name):
    if name == "Iris": data = datasets.load_iris()
    elif name == "Wine": data = datasets.load_wine()
    else: data = datasets.load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target, data.target_names

df_x, y, target_names = load_data(dataset_name)
st.write(f"Dataset seleccionado: **{dataset_name}** ({df_x.shape[0]} muestras, {df_x.shape[1]} características)")

# --- PREPROCESAMIENTO Y PCA ---
st.sidebar.header("2. Preprocesamiento")
do_pca = st.sidebar.checkbox("¿Activar PCA?")
variance_threshold = st.sidebar.slider("Varianza explicada deseada", 0.50, 0.99, 0.95) if do_pca else 1.0

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_x)

if do_pca:
    pca_obj = PCA(n_components=variance_threshold)
    X_final = pca_obj.fit_transform(X_scaled)
    st.sidebar.info(f"Componentes creados: {pca_obj.n_components_}")
else:
    X_final = X_scaled

# --- MODELO Y MÉTRICAS ---
classifier_name = st.sidebar.selectbox("3. Clasificador", ("KNN", "Decision Tree", "Naive Bayes", "LDA"))
metrics_selected = st.sidebar.multiselect("4. Métricas a evaluar", 
                                         ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                                         default=["accuracy", "f1_macro"])

# --- VALIDACIÓN CRUZADA ---
st.sidebar.header("5. Validación")
cv_folds = st.sidebar.number_input("Número de Folds (K-Fold)", 2, 10, 5)

# Instanciar modelo
def get_model(name):
    if name == "KNN": return KNeighborsClassifier()
    if name == "Decision Tree": return DecisionTreeClassifier()
    if name == "Naive Bayes": return GaussianNB()
    return LDA()

model = get_model(classifier_name)

# --- EJECUCIÓN ---
if st.button("🚀 Entrenar y Evaluar"):
    # 1. Validación Cruzada
    cv_results = cross_validate(model, X_final, y, cv=KFold(n_splits=cv_folds, shuffle=True), 
                                scoring=metrics_selected, return_train_score=False)
    
    # 2. Split simple para visualización de matriz
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # --- RESULTADOS ---
    st.header("📊 Resultados del Desempeño")
    
    cols = st.columns(len(metrics_selected))
    for i, m in enumerate(metrics_selected):
        mean_score = np.mean(cv_results[f'test_{m}'])
        cols[i].metric(m.upper(), f"{mean_score:.3f}")

    # Notificación de Despliegue
    main_metric = np.mean(cv_results[f'test_{metrics_selected[0]}'])
    if main_metric > 0.85:
        st.success(f"🌟 ¡Modelo de alto rendimiento! Listo para despliegue (Promedio: {main_metric:.2f})")
    
    # --- VISUALIZACIONES ---
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Matriz de Confusión")
        fig_cm, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=target_names, yticklabels=target_names)
        st.pyplot(fig_cm)

    with col_right:
        st.subheader("Distribución de Clases (Predicción)")
        res_df = pd.DataFrame({'Clase': [target_names[i] for i in y_pred]})
        fig_pie = px.pie(res_df, names='Clase', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie)

    # Gráfica de Varianza de PCA si aplica
    if do_pca:
        st.subheader("Análisis de Varianza Acumulada (PCA)")
        acc_var = np.cumsum(pca_obj.explained_variance_ratio_)
        fig_pca = px.line(x=range(1, len(acc_var)+1), y=acc_var, labels={'x': 'Componentes', 'y': 'Varianza'})
        st.plotly_chart(fig_pca)
