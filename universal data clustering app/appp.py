import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Streamlit Page Config
st.set_page_config(page_title="Universal Data Clustering", page_icon="📊", layout="wide")

# CSS for better UI
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 18px;
        }
        .stFileUploader {
            border: 2px dashed #4CAF50;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Universal Data Clustering App")

# Upload File
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.write(df.head())

    # Identify categorical & numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric missing values with median
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical missing values with mode

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # User selects features for clustering
    st.subheader("🛠 Select Features for Clustering")
    selected_features = st.multiselect("Choose features:", numerical_cols + categorical_cols, default=numerical_cols)

    if selected_features:
        # Scaling selected features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[selected_features])

        # Find optimal number of clusters (Elbow Method)
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            wcss.append(kmeans.inertia_)

        # Plot Elbow Method
        st.subheader("📊 Elbow Method for Optimal Clusters")
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig)

        # Select number of clusters
        k = st.slider("🔢 Select Number of Clusters", min_value=2, max_value=10, value=4, step=1)

        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_features)

        # PCA for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_features)
        df['PCA1'] = reduced_data[:, 0]
        df['PCA2'] = reduced_data[:, 1]

        # Display Cluster Distribution
        st.subheader("📌 Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        st.table(cluster_counts)

        # Visualize Clusters
        st.subheader("🎨 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette="viridis", s=100, alpha=0.7)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("Data Clusters")
        st.pyplot(fig)

        # Download clustered dataset
        st.subheader("📥 Download Clustered Dataset")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV", data=csv, file_name="Clustered_Data.csv", mime='text/csv')

        st.success("✅ Clustering completed! Download the clustered dataset above.")
else:
    st.info("Please upload a CSV dataset to proceed.")
