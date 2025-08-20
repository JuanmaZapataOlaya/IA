import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="ML Supervisado y No Supervisado", layout="wide")

st.title("🧠 Modelos de Aprendizaje Supervisado y No Supervisado")

# Cargar datos
st.sidebar.header("Carga tus datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Vista previa del dataset", df.head())

    if st.sidebar.checkbox("Mostrar descripción del dataset"):
        st.write(df.describe())

    st.sidebar.subheader("Selecciona el tipo de modelo")
    model_type = st.sidebar.radio("Tipo de aprendizaje", ("Supervisado", "No Supervisado"))

    if model_type == "Supervisado":
        target_column = st.sidebar.selectbox("Selecciona la columna objetivo", df.columns)
        features = df.drop(columns=[target_column])
        labels = df[target_column]

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

        st.sidebar.subheader("Modelo Supervisado")
        model_choice = st.sidebar.selectbox("Modelo", ["Logistic Regression", "Random Forest"])

        if st.sidebar.button("Entrenar modelo"):
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.subheader("Reporte de Clasificación")
            st.text(classification_report(y_test, predictions))

            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    elif model_type == "No Supervisado":
        st.sidebar.subheader("Modelo No Supervisado")
        model_choice = st.sidebar.selectbox("Modelo", ["KMeans", "PCA"])

        n_components = st.sidebar.slider("Número de Clusters / Componentes", 2, 10, 3)

        if st.sidebar.button("Ejecutar modelo"):
            features = df.select_dtypes(include=["float64", "int64"])

            if model_choice == "KMeans":
                model = KMeans(n_clusters=n_components, random_state=42)
                clusters = model.fit_predict(features)
                df["Cluster"] = clusters
                st.subheader("Clusters asignados")
                st.write(df.head())

                fig, ax = plt.subplots()
                sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], hue=clusters, palette="Set2", ax=ax)
                st.pyplot(fig)

            elif model_choice == "PCA":
                model = PCA(n_components=n_components)
                components = model.fit_transform(features)

                st.subheader("Componentes Principales")
                comp_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)])
                st.write(comp_df.head())

                if n_components >= 2:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=comp_df["PC1"], y=comp_df["PC2"], ax=ax)
                    st.pyplot(fig)
else:
    st.warning("Por favor, sube un archivo CSV para comenzar.")
