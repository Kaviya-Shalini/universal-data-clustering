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
st.set_page_config(page_title="Customer Segmentation - TapToBuy", page_icon="🛒", layout="wide")

# Attractive UI Styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .css-1d391kg {
            background-color: #ffffff;
        }
        .stButton>button {
            background-color: #ff5733;
            color: white;
            border-radius: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🛍️ TapToBuy Customer Segmentation")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload TapToBuy CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Overview")
    st.write(df.head())

    # Data Preprocessing
    st.subheader("⚙️ Data Preprocessing")

    # Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")
    df[['Ever_Married', 'Graduated', 'Profession']] = imputer.fit_transform(df[['Ever_Married', 'Graduated', 'Profession']])
    df[['Work_Experience', 'Family_Size']] = df[['Work_Experience', 'Family_Size']].fillna(df[['Work_Experience', 'Family_Size']].median())

    # Encode categorical variables
    encoder = LabelEncoder()
    for col in ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']:
        df[col] = encoder.fit_transform(df[col])

    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=['ID']))

    # Elbow Method for optimal clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    # Plot Elbow Method
    st.subheader("📊 Elbow Method to Determine Clusters")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method for Optimal k")
    st.pyplot(fig)

    # User selects number of clusters
    k = st.slider("🔢 Select Number of Clusters (k)", min_value=2, max_value=10, value=4, step=1)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_features)
    df['PCA1'] = reduced_data[:, 0]
    df['PCA2'] = reduced_data[:, 1]

    # Show Cluster Distribution
    st.subheader("📌 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    st.table(cluster_counts)

    # Visualize Clusters
    st.subheader("🎨 Customer Segments Visualization")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette="viridis", s=100, alpha=0.7)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Customer Clusters")
    st.pyplot(fig)

    # Download clustered dataset
    st.subheader("📥 Download Clustered Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download CSV", data=csv, file_name="Clustered_TapToBuy.csv", mime='text/csv')

    st.success("✅ Clustering completed! Download the clustered dataset above.")
else:
    st.info("Please upload the TapToBuy dataset to proceed.")
