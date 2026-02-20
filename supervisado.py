import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(page_title="AutoML Classifier", layout="wide")

st.title("🤖 Clasificador Inteligente con Streamlit")
st.markdown("""
Esta app carga el dataset **Iris**, preprocesa los datos, permite extraer características 
y evalúa si el modelo es apto para despliegue.
""")

# --- 1. CARGA DE DATOS ---
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

st.sidebar.header("Configuración del Modelo")

# --- 2. PREPROCESO Y 3. FEATURE EXTRACTION ---
st.subheader("1. Análisis y Preprocesamiento")
use_pca = st.sidebar.checkbox("¿Aplicar PCA (Feature Extraction)?", value=False)

X = df.drop("target", axis=1)
y = df["target"]

if use_pca:
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    st.info("Se ha reducido la dimensionalidad a 2 componentes principales.")
else:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    st.info("Datos normalizados con StandardScaler.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. SELECCIÓN DE MODELO ---
classifier_name = st.sidebar.selectbox(
    "Selecciona el Clasificador",
    ("KNN", "Decision Tree", "Naive Bayes", "LDA")
)

def get_classifier(name):
    if name == "KNN":
        k = st.sidebar.slider("K (vecinos)", 1, 15, 3)
        return KNeighborsClassifier(n_neighbors=k)
    elif name == "Decision Tree":
        max_d = st.sidebar.slider("Profundidad máxima", 1, 10, 5)
        return DecisionTreeClassifier(max_depth=max_d)
    elif name == "Naive Bayes":
        return GaussianNB()
    else:
        return LDA()

clf = get_classifier(classifier_name)

# --- 5. VALIDACIÓN ---
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- 6. UMBRAL DE DESPLIEGUE ---
st.subheader(f"Resultados: {classifier_name}")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy (Precisión)", f"{acc*100:.2f}%")
    
    if acc >= 0.90:
        st.success("✅ ¡Desempeño Alto! El modelo está listo para producción.")
        # Aquí iría la lógica de guardado de modelo (joblib/pickle)
    elif acc >= 0.75:
        st.warning("⚠️ Desempeño Medio. Se recomienda ajustar hiperparámetros.")
    else:
        st.error("❌ Desempeño Bajo. No apto para despliegue.")

# --- 7. GRÁFICAS DE DESEMPEÑO ---
with col2:
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    st.pyplot(fig)

with st.expander("Ver reporte detallado"):
    st.text(classification_report(y_test, y_pred, target_names=target_names))
