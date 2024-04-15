import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
sns.set_theme(color_codes=True)
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import streamlit as st

st.title("Ecommerce Segmentation Analysis")
    
# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV file:")

# Check if the file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Show the DataFrame
    st.dataframe(df)

# Get numeric columns for clustering
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    clustering_columns = st.multiselect("Select numeric columns for clustering:", numeric_columns)

    # Check if at least 3 columns are selected
    if len(clustering_columns) != 3:
        st.warning("Please select exactly 3 numeric columns for clustering.")
    else:
        # Display the selected columns
        st.subheader("Selected Columns for Clustering:")
        selected_data = df[clustering_columns]
        st.dataframe(selected_data)

        # Remove missing values
        selected_data.dropna(inplace=True)

        def visualize_clustering(df, selected_data):
            # Visualize the Elbow Method to find optimal clusters
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(selected_data)
                wcss.append(kmeans.inertia_)

            # Plot the Elbow Method
            st.subheader("Elbow Method to Determine Optimal Clusters")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(1, 11), wcss, marker='o')
            ax.set_title('Elbow Method')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('WCSS')  # Within-Cluster Sum of Squares
            st.pyplot(fig)

            # Visualize Silhouette Score for different cluster numbers
            silhouette_scores = []
            for n_clusters in range(2, 11):
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(selected_data)
                silhouette_avg = silhouette_score(selected_data, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)
                
            # Plot Silhouette Score
            st.subheader("Silhouette Score for Different Cluster Numbers")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(2, 11), silhouette_scores, marker='o')
            ax.set_title('Silhouette Score')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Silhouette Score')
            st.pyplot(fig)

            # Apply KMeans clustering based on user-selected number of clusters
            num_clusters = st.slider("Select the number of clusters (2-10):", 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            cluster_labels = kmeans.fit_predict(selected_data)

            # Create a new DataFrame with the cluster labels
            clustered_df = pd.DataFrame(cluster_labels, columns=['cluster'], index=selected_data.index)

            # Concatenate the clustered_df with the original DataFrame
            df = pd.concat([df, clustered_df], axis=1)
            st.subheader("Clustered Dataset")
            st.dataframe(df)

            # Visualize clustering results in 3D plot
            fig = plt.figure(figsize=(10, 12))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(selected_data[clustering_columns[0]], 
                                selected_data[clustering_columns[1]], 
                                selected_data[clustering_columns[2]], 
                                c=cluster_labels, cmap='viridis', s=50)
                
            ax.set_xlabel(clustering_columns[0])
            ax.set_ylabel(clustering_columns[1])
            ax.set_zlabel(clustering_columns[2])
            ax.set_title(f'3D Clustering (Cluster Amount = {num_clusters})')

            # Add a legend
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)

            # Show the 3D plot
            st.pyplot(fig)
            fig.savefig("plot8.png")

        # Visualize clustering
        visualize_clustering(df, selected_data)


        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyCY-mXpPt-J0oGRaSiPaeAyAVollbMxCF8")

        import PIL.Image

        img = PIL.Image.open("plot8.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the each cluster colour. write the conclusion in English", img], stream=True)
        response.resolve()
        st.subheader("**Google Gemini Response About Data**")
        st.write(response.text)



