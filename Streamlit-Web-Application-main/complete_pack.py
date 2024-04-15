import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PyPDF2
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import warnings
sns.set_theme(color_codes=True)
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

# Classification Prediction
def page1():
    st.title("Classification Prediction")
    
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV file:")

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Show the DataFrame
        st.dataframe(df)

        #Check the number of unique value from all of the object datatype
        st.write("Amount of Unique Value on Column with Object Datatype")
        nunique = df.select_dtypes(include='object').nunique()
        st.write(nunique)

        st.write("Amount of missing value in percentage :")
        # Print the amount of missing value
        check_missing = df.isnull().sum() * 100 / df.shape[0]
        missing = check_missing[check_missing > 0].sort_values(ascending=False)
        st.write(missing)

        # Drop all of the column where null value > 20%
        st.write("Drop all of the column where the missing value more than 20%")
        columns_to_remove = missing[missing > 20].index
        df = df.drop(columns=columns_to_remove)
        st.dataframe(df)

        # Drop all of the null value for all of the "Object" datatype
        st.write("Drop All of the null value on Object Column")
        df = df.dropna(subset=df.select_dtypes(include=['object']).columns)
        st.dataframe(df)

        # Remove Selected Columns
        target_variables = st.multiselect("Select columns to remove", df.columns)
        df.drop(columns = target_variables, inplace=True)
        st.dataframe(df)

        # Select EDA Method
        select_method = st.selectbox('Select your filling null value method for numeric column', ("Fill with Mean", "Fill with Median"))

        if select_method == "Fill with Mean":
            # Fill null values in float or integer columns with more than 10 unique values with mean
            numeric_columns = df.select_dtypes(include=['float', 'int'])
            unique_value_counts = df[numeric_columns.columns].nunique()
            columns_to_fill = unique_value_counts[unique_value_counts > 10].index
            df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].mean())
            st.dataframe(df)

        elif select_method == "Fill with Median":
            # Fill null values in float or integer columns with more than 10 unique values with median
            numeric_columns = df.select_dtypes(include=['float', 'int'])
            unique_value_counts = df[numeric_columns.columns].nunique()
            columns_to_fill = unique_value_counts[unique_value_counts > 10].index
            df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())
            st.dataframe(df)
    
        # Drop columns where unique values are greater than 30 in object columns
        st.write("Drop columns where unique values > 30 in object columns")

        object_columns = df.select_dtypes(include=['object'])
        unique_value_counts = df[object_columns.columns].nunique()
        columns_to_drop = unique_value_counts[unique_value_counts > 30].index

        # Drop the selected columns
        df = df.drop(columns=columns_to_drop)

        st.dataframe(df)


        # Label Encoding for Object Datatypes
        # Loop over each column in the DataFrame where dtype is 'object'
        st.write("**Show all of the Unique Value on all of Object Datatype**")
        for col in df.select_dtypes(include=['object']).columns:
    
            # Print the column name and the unique values
            st.write(f"{col}: {df[col].unique()}")
        st.write("")

        # Loop over each column in the DataFrame where dtype is 'object'
        st.write("**Label Encooding for all of the column with Object Datatype**")
        from sklearn import preprocessing
        for col in df.select_dtypes(include=['object']).columns:
    
            # Initialize a LabelEncoder object
            label_encoder = preprocessing.LabelEncoder()
    
            # Fit the encoder to the unique values in the column
            label_encoder.fit(df[col].unique())
    
            # Transform the column using the encoder
            df[col] = label_encoder.transform(df[col])
    
            # Print the column name and the unique encoded values
            st.write(f"{col}: {df[col].unique()}")
        st.write("")

        # Display Correlation Heatmap
        st.write("**Show the heatmap correlation**")
        fig, ax = plt.subplots(figsize=(30, 24))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Train Test Split
        st.write("**Train Test Split**")
        cols = df.columns.tolist()
        target_variable = st.selectbox('Select a column:', cols)
        number = st.slider("Choose Test Size Percentage", 0, 100)
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

        #Remove Outliers on numerical column
        st.write("**Remove Outlier from selected columns using Z-Score**")
        cat_cols = df.columns.tolist()
        target_variables = st.multiselect("Select columns to remove the outlier", df.columns)

        # Calculate the Z-scores for the selected columns in the training data
        z_scores = np.abs(stats.zscore(X_train[target_variables]))

        # Set a threshold value for outlier detection (e.g., 3)
        threshold = 3

        # Find the indices of outliers based on the threshold
        outlier_indices = np.where(z_scores > threshold)[0]

        # Remove the outliers from the training data
        X_train = X_train.drop(X_train.index[outlier_indices])
        y_train = y_train.drop(y_train.index[outlier_indices])

        st.title("Machine Learning Modelling")

        select_method2a = st.selectbox('Select your Machine Learning Classification Algorithm', ("Decision Tree Classifier", "Random Forest Classifier"))
        if select_method2a == "Decision Tree Classifier":
            rs1 = st.selectbox('Random State', (0, 42))
            md1 = st.selectbox('Max Depth', (3, 4, 5, 6, 7, 8))
            msl1 = st.selectbox('Min Sample Leaf', (1, 2, 3, 4))
            mss1 = st.selectbox('Min Sample Split', (2, 3, 4))
            from sklearn.tree import DecisionTreeClassifier
            dtree = DecisionTreeClassifier(random_state=rs1, max_depth=md1, min_samples_leaf=msl1, min_samples_split=mss1, class_weight='balanced')
            dtree.fit(X_train, y_train)

            from sklearn.metrics import accuracy_score
            y_pred = dtree.predict(X_test)
            st.write("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
            st.write('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
            st.write('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
            st.write('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
            st.write('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            # Create the barplot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Deicision Tree Classifier)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)

            from sklearn.metrics import confusion_matrix            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Accuracy Score for Decision Tree: {0:.2f} %'.format(dtree.score(X_test, y_test)*100)
            plt.title(all_sample_title, size = 15)
            st.pyplot(fig)
        
        if select_method2a == "Random Forest Classifier":
            rs2 = st.selectbox('Random State', (0, 42))
            md2 = st.selectbox('Max Depth', (3, 4, 5, 6, 7, 8))
            mf2 = st.selectbox('Max Features', ('sqrt', 'log2', None))
            ne2 = st.selectbox('N Estimator', (100, 200))
            from sklearn.ensemble import RandomForestClassifier
            rfc = RandomForestClassifier(random_state=rs2, max_depth=md2, max_features=mf2, n_estimators=ne2, class_weight='balanced')
            rfc.fit(X_train, y_train)

            from sklearn.metrics import accuracy_score
            y_pred = rfc.predict(X_test)
            st.write("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss
            st.write('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
            st.write('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
            st.write('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
            st.write('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rfc.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            # Create the barplot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Random Forest Classifier)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)

            from sklearn.metrics import confusion_matrix            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            all_sample_title = 'Accuracy Score for Random Forest: {0:.2f} %'.format(rfc.score(X_test, y_test)*100)
            plt.title(all_sample_title, size = 15)
            st.pyplot(fig)


# Regression Prediction
def page2():
    st.title("Regression Prediction")
    
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV file:")

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Show the DataFrame
        st.dataframe(df)

        #Check the number of unique value from all of the object datatype
        st.write("Amount of Unique Value on Column with Object Datatype")
        nunique = df.select_dtypes(include='object').nunique()
        st.write(nunique)

        st.write("Amount of missing value in percentage :")
        # Print the amount of missing value
        check_missing = df.isnull().sum() * 100 / df.shape[0]
        missing = check_missing[check_missing > 0].sort_values(ascending=False)
        st.write(missing)

        # Drop all of the column where null value > 20%
        st.write("Drop all of the column where the missing value more than 20%")
        columns_to_remove = missing[missing > 20].index
        df = df.drop(columns=columns_to_remove)
        st.dataframe(df)

        # Drop all of the null value for all of the "Object" datatype
        st.write("Drop All of the null value on Object Column")
        df = df.dropna(subset=df.select_dtypes(include=['object']).columns)
        st.dataframe(df)

        # Remove Selected Columns
        target_variables = st.multiselect("Select columns to remove", df.columns)
        df.drop(columns = target_variables, inplace=True)
        st.dataframe(df)

        # Select EDA Method
        select_method = st.selectbox('Select your filling null value method for numeric column', ("Fill with Mean", "Fill with Median"))

        if select_method == "Fill with Mean":
            # Fill null values in float or integer columns with more than 10 unique values with mean
            numeric_columns = df.select_dtypes(include=['float', 'int'])
            unique_value_counts = df[numeric_columns.columns].nunique()
            columns_to_fill = unique_value_counts[unique_value_counts > 10].index
            df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].mean())
            st.dataframe(df)

        elif select_method == "Fill with Median":
            # Fill null values in float or integer columns with more than 10 unique values with median
            numeric_columns = df.select_dtypes(include=['float', 'int'])
            unique_value_counts = df[numeric_columns.columns].nunique()
            columns_to_fill = unique_value_counts[unique_value_counts > 10].index
            df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())
            st.dataframe(df)
    
        # Drop columns where unique values are greater than 30 in object columns
        st.write("Drop columns where unique values > 30 in object columns")

        object_columns = df.select_dtypes(include=['object'])
        unique_value_counts = df[object_columns.columns].nunique()
        columns_to_drop = unique_value_counts[unique_value_counts > 30].index

        # Drop the selected columns
        df = df.drop(columns=columns_to_drop)

        st.dataframe(df)


        # Label Encoding for Object Datatypes
        # Loop over each column in the DataFrame where dtype is 'object'
        st.write("**Show all of the Unique Value on all of Object Datatype**")
        for col in df.select_dtypes(include=['object']).columns:
    
            # Print the column name and the unique values
            st.write(f"{col}: {df[col].unique()}")
        st.write("")

        # Loop over each column in the DataFrame where dtype is 'object'
        st.write("**Label Encooding for all of the column with Object Datatype**")
        from sklearn import preprocessing
        for col in df.select_dtypes(include=['object']).columns:
    
            # Initialize a LabelEncoder object
            label_encoder = preprocessing.LabelEncoder()
    
            # Fit the encoder to the unique values in the column
            label_encoder.fit(df[col].unique())
    
            # Transform the column using the encoder
            df[col] = label_encoder.transform(df[col])
    
            # Print the column name and the unique encoded values
            st.write(f"{col}: {df[col].unique()}")
        st.write("")

        # Display Correlation Heatmap
        st.write("**Show the heatmap correlation**")
        fig, ax = plt.subplots(figsize=(30, 24))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Train Test Split
        st.write("**Train Test Split**")
        cols = df.columns.tolist()
        target_variable = st.selectbox('Select a column:', cols)
        number = st.slider("Choose Test Size Percentage", 0, 100)
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

        #Remove Outliers on numerical column
        st.write("**Remove Outlier from selected columns using Z-Score**")
        cat_cols = df.columns.tolist()
        target_variables = st.multiselect("Select columns to remove the outlier", df.columns)

        # Calculate the Z-scores for the selected columns in the training data
        z_scores = np.abs(stats.zscore(X_train[target_variables]))

        # Set a threshold value for outlier detection (e.g., 3)
        threshold = 3

        # Find the indices of outliers based on the threshold
        outlier_indices = np.where(z_scores > threshold)[0]

        # Remove the outliers from the training data
        X_train = X_train.drop(X_train.index[outlier_indices])
        y_train = y_train.drop(y_train.index[outlier_indices])

        select_method2a = st.selectbox('Select your Machine Learning Regressor Algorithm', ("Decision Tree Regressor", "Random Forest Regressor"))
        if select_method2a == "Decision Tree Regressor":
            rs1 = st.selectbox('Random State', (0, 42))
            md1 = st.selectbox('Max Depth', (3, 4, 5, 6, 7, 8))
            mf1 = st.selectbox('Max Features', ('sqrt', 'log2'))
            mss1 = st.selectbox('Min Sample Split', (2, 4, 6, 8))
            msl1 = st.selectbox('Min Sample Leaf', (1, 2, 3, 4))
            from sklearn.tree import DecisionTreeRegressor
            dtree = DecisionTreeRegressor(random_state=rs1, max_depth=md1, max_features=mf1, min_samples_leaf=msl1, min_samples_split=mss1)
            dtree.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import mean_absolute_percentage_error
            import math
            y_pred = dtree.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = math.sqrt(mse)

            st.write('MAE is {}'.format(mae))
            st.write('MAPE is {}'.format(mape))
            st.write('MSE is {}'.format(mse))
            st.write('R2 score is {}'.format(r2))
            st.write('RMSE score is {}'.format(rmse))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Decision Tree Regressor)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)
        
        if select_method2a == "Random Forest Regressor":
            rs1 = st.selectbox('Random State', (0, 42))
            md1 = st.selectbox('Max Depth', (3, 5, 7, 9))
            mf1 = st.selectbox('Max Features', ('log2', 'sqrt'))
            mss1 = st.selectbox('Min Sample Split', (2, 5, 10))
            msl1 = st.selectbox('Min Sample Leaf', (1, 2, 4))
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(random_state=rs1, max_depth=md1, min_samples_split=mss1, min_samples_leaf=msl1, max_features=mf1)
            rf.fit(X_train, y_train)

            from sklearn import metrics
            from sklearn.metrics import mean_absolute_percentage_error
            import math
            y_pred = rf.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = math.sqrt(mse)

            st.write('MAE is {}'.format(mae))
            st.write('MAPE is {}'.format(mape))
            st.write('MSE is {}'.format(mse))
            st.write('R2 score is {}'.format(r2))
            st.write('RMSE score is {}'.format(rmse))

            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rf.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False)

            fi2 = fi.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi2, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance Each Attributes (Random Forest Regressor)', fontsize=18)
            ax.set_xlabel('Importance', fontsize=16)
            ax.set_ylabel('Feature Name', fontsize=16)

            # Display the plot in Streamlit
            st.pyplot(fig)

# NLP : PDF Document Analysis
def page3():
    st.title("NLP : PDF Document Analysis")

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from wordcloud import WordCloud
    import PyPDF2
    import re
    from io import StringIO
    import plotly.express as px
    import pandas as pd
    import collections

    # Create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Create stopword remover
    stop_factory = StopWordRemoverFactory()
    more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
    data = stop_factory.get_stop_words() + more_stopword

    # User input for custom stopwords
    custom_stopwords = st.text_input("Enter custom stopwords (comma-separated):")
    if custom_stopwords:
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")]
        data.extend(custom_stopword_list)

    # Function to read PDF and return string
    def read_pdf(file):
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extractText()
        return text

    # Upload PDF file
    file = st.file_uploader("Upload a PDF file", type="pdf", key='text1')

    # If file is uploaded
    if file is not None:
        # Call read_pdf function to convert PDF to string
        text1 = read_pdf(file)

        # Stem and preprocess the text
        sentence1 = text1
        output1 = stemmer.stem(sentence1)
        hasil1 = re.sub(r"\d+", "", output1)
        hasil1 = re.sub(r'[^a-zA-Z\s]', '', hasil1)
        pattern = re.compile(r'\b(' + r'|'.join(data) + r')\b\s*')
        hasil1 = pattern.sub('', hasil1)

        # Create WordCloud
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(hasil1)

        # Display the WordCloud using Streamlit
        st.write("**WordCloud Visualization**")
        st.image(wordcloud.to_array())

        # Bigram visualization
        # Get bigrams
        words1 = hasil1.split()
        bigrams = list(zip(words1, words1[1:]))

        # Count bigrams
        bigram_counts = collections.Counter(bigrams)

        # Get top 10 bigram counts
        top_bigrams = dict(bigram_counts.most_common(10))

        # Sort the data for standard bar plot (without Plotly)
        sorted_bigrams = sorted(top_bigrams.items(), key=lambda x: x[1], reverse=True)

        # Create standard bar plot
        st.write("**Bi-Gram Analysis Visualization**")
        st.bar_chart(pd.DataFrame(sorted_bigrams, columns=["Bigram Words", "Count"]).set_index("Bigram Words"))

# Sentiment Analysis
def page4():
    st.title("Sentiment Analysis")

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from wordcloud import WordCloud
    import PyPDF2
    import re
    from io import StringIO
    import plotly.express as px
    import pandas as pd
    import collections
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    # Create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Create stopword remover
    stop_factory = StopWordRemoverFactory()
    more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
    data = stop_factory.get_stop_words() + more_stopword

    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV file:")

    # User input for delimiter
    delimiter_option = st.radio("Select CSV delimiter:", [",", ";"], index=0)

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        if delimiter_option == ",":
            df = pd.read_csv(uploaded_file, delimiter=",")
        elif delimiter_option == ";":
            df = pd.read_csv(uploaded_file, delimiter=";")
        else:
            st.error("Invalid delimiter option.")

        # Show the DataFrame
        st.dataframe(df)

        # Select a column for sentiment analysis
        object_columns = df.select_dtypes(include="object").columns
        target_variable = st.selectbox("Choose a column for Sentiment Analysis:", object_columns)

        # Perform sentiment analysis on the selected column
        if st.button("Perform Sentiment Analysis"):
            # Your sentiment analysis logic goes here
            st.success(f"Sentiment Analysis performed on column: {target_variable}")
        
        # Show the selected column
        st.write(f"Selected {target_variable} Column:")
        st.dataframe(df[[target_variable]])

        # Create a new DataFrame with cleaned text column
        new_df = df.copy()

        # Apply stemming and stopword removal to the selected column
        new_df['cleaned_text'] = new_df[target_variable].apply(lambda x: ' '.join([stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split() if word.lower() not in data]))

        # Show the cleaned text column
        st.write("Cleaned Text Column:")
        st.dataframe(new_df[['cleaned_text']])

        # Load the sentiment analysis pipeline
        pretrained = "indonesia-bert-sentiment-classification"
        model = AutoModelForSequenceClassification.from_pretrained(pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

        # Function to apply sentiment analysis to each row in the 'cleaned_text' column
        def analyze_sentiment(text):
            result = sentiment_analysis(text)
            label = label_index[result[0]['label']]
            score = result[0]['score']
            return pd.Series({'sentiment_label': label, 'sentiment_score': score})

        # Apply sentiment analysis to 'cleaned_text' column
        new_df[['sentiment_label', 'sentiment_score']] = new_df['cleaned_text'].apply(analyze_sentiment)

        # Display the results
        st.write("Sentiment Analysis Results:")
        st.dataframe(new_df[['cleaned_text', 'sentiment_label', 'sentiment_score']])

        # Count the occurrences of each sentiment label
        sentiment_counts = new_df['sentiment_label'].value_counts()

        # Plot a bar chart using seaborn
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        st.pyplot()

        # Choose sentiment and display corresponding data
        chosen_sentiment = st.selectbox("Choose a sentiment:", sentiment_counts.index)
        st.subheader(f"Data for Sentiment: {chosen_sentiment}")
        selected_data = new_df[new_df['sentiment_label'] == chosen_sentiment]
        st.write(selected_data)

        # Display WordCloud for the selected sentiment
        st.subheader(f"WordCloud for Sentiment: {chosen_sentiment}")
        sentiment_text = ' '.join(selected_data['cleaned_text'].astype(str))

        # Add custom stopwords
        custom_stopwords = st.text_input("Enter custom stopwords (comma-separated):")
        if custom_stopwords:
            custom_stopword_list = [word.strip() for word in custom_stopwords.split(",")]
            sentiment_text = ' '.join([word for word in sentiment_text.split() if word.lower() not in custom_stopword_list])

        # Create WordCloud
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(sentiment_text)

        # Display the WordCloud
        st.image(wordcloud.to_array())

        # Display top 10 bigrams
        from sklearn.feature_extraction.text import CountVectorizer
        import collections
        # Sample input
        text = sentiment_text
        words = text.split()

        # Get bigrams
        bigrams = list(zip(words, words[1:]))

        # Count bigrams
        bigram_counts = collections.Counter(bigrams)

        # Get top 10 bigram counts
        top_bigrams = dict(bigram_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()))
        plt.xticks(rotation=90)
        plt.xlabel('Bigram Words')
        plt.ylabel('Count')
        plt.title(f"Top 10 Bigram from {chosen_sentiment}")
        st.subheader(f"Bigram for Sentiment: {chosen_sentiment}")
        st.pyplot()

def page5():
    st.title("K-means Clustering")
    
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
                fig = plt.figure(figsize=(10, 8))
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

            # Visualize clustering
            visualize_clustering(df, selected_data)

def page6():
    st.title("Exploratory Data Analysis")
    
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV file:")

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Show the original DataFrame
        st.write("Original DataFrame:")
        st.dataframe(df)
        
        # Select target variable
        target_variable = st.selectbox("Select target variable:", df.columns)
        
        # Select columns for analysis
        columns_for_analysis = st.multiselect("Select columns for analysis:", [col for col in df.columns if col != target_variable])

        # Process button
        if st.button("Process"):
            # Select the target variable and columns for analysis from the original DataFrame
            target_variable_data = df[target_variable]
            columns_for_analysis_data = df[columns_for_analysis]

            # Display target variable in a dataframe
            target_variable_df = df[[target_variable]]
            st.write("Target Variable DataFrame:")
            st.dataframe(target_variable_df)
            
            # Display columns for analysis in a dataframe
            columns_for_analysis_df = df[columns_for_analysis]
            st.write("Columns for Analysis DataFrame:")
            st.dataframe(columns_for_analysis_df)
            
            # Concatenate target variable and columns for analysis into a single DataFrame
            df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)
            st.write("Columns for Analysis and Target Variable DataFrame:")
            st.dataframe(df)

            # Select EDA Method
            st.write("Countplot Barchart")

            # Get the names of all columns with data type 'object' (categorical columns) excluding 'Country'
            cat_vars = df.select_dtypes(include='object').columns.tolist()

            # Create a figure with subplots
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a countplot for the top 10 values of each categorical variable using Seaborn
            for i, var in enumerate(cat_vars):
                top_values = df[var].value_counts().head(10).index
                filtered_df = df.copy()
                filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
                sns.countplot(x=var, data=filtered_df, ax=axs[i])
                axs[i].set_title(var)
                axs[i].tick_params(axis='x', rotation=90)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plots using Streamlit
            st.pyplot(fig)
        

            st.write("Pie Chart")
            # Specify the maximum number of categories to show individually
            max_categories = 5

            # Filter categorical columns with 'object' data type
            cat_cols = [col for col in df.columns if col != 'y' and df[col].dtype == 'object']

            # Create a figure with subplots
            num_cols = len(cat_cols)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(20, 5*num_rows))
            axs = axs.flatten()

            # Create a pie chart for each categorical column
            for i, col in enumerate(cat_cols):
                if i < len(axs):  # Ensure we don't exceed the number of subplots
                    # Count the number of occurrences for each category
                    cat_counts = df[col].value_counts()

                    # Group categories beyond the top max_categories as 'Other'
                    if len(cat_counts) > max_categories:
                        top_categories = cat_counts.head(max_categories)
                        other_category = pd.Series(cat_counts[max_categories:].sum(), index=['Other'])
                        cat_counts = pd.concat([top_categories, other_category])

                    # Create a pie chart
                    axs[i].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
                    axs[i].set_title(f'{col} Distribution')

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Display plots using Streamlit
            st.pyplot(fig)

    
            st.write("Box Plot")
            # Get the names of all columns with data type 'int' or 'float', excluding 'churn_risk_score'
            num_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()

            # Create a figure with subplots
            num_cols = len(num_vars)
            num_rows = (num_cols + 2) // 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a box plot for each numerical variable using Seaborn
            for i, var in enumerate(num_vars):
                sns.boxplot(x=df[var], ax=axs[i])
                axs[i].set_title(var)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Display the boxplot using Streamlit
            st.pyplot(fig)


            st.write("Histoplot")
            # Get the names of all columns with data type 'int'
            int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()

            # Create a figure with subplots
            num_cols = len(int_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a histogram for each integer variable
            for i, var in enumerate(int_vars):
                df[var].plot.hist(ax=axs[i])
                axs[i].set_title(var)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)


            st.write("Multi Class Boxplot")
            # Get the names of all columns with data type 'object' (categorical columns)
            cat_cols = df.columns.tolist()

            num_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
            # Select a categorical column
            int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
            int_vars = [col for col in num_vars if col != target_variable]

            # Create a figure with subplots
            num_cols = len(int_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()
            # Create a box plot for each integer variable using Seaborn with hue='attrition'
            for i, var in enumerate(int_vars):
                sns.boxplot(y=var, x=target_variable, data=df, ax=axs[i])
                axs[i].set_title(var)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)



            st.write("Multi Class Density Plot")
            cat_cols = df.columns.tolist()

            # Get the names of all columns with data type 'object' (categorical variables)
            cat_vars = df.select_dtypes(include=['object']).columns.tolist()

            # Exclude 'Attrition' from the list if it exists in cat_vars
            if  target_variable in cat_vars:
                cat_vars.remove( target_variable)

            # Create a figure with subplots, but only include the required number of subplots
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a count plot for the top 6 values of each categorical variable as a density plot
            for i, var in enumerate(cat_vars):
                top_values = df[var].value_counts().nlargest(6).index
                filtered_df = df[df[var].isin(top_values)]
        
                # Set x-tick positions explicitly
                tick_positions = range(len(top_values))
                axs[i].set_xticks(tick_positions)
                axs[i].set_xticklabels(top_values, rotation=90)  # Set x-tick labels
        
                sns.histplot(x=var, hue= target_variable, data=filtered_df, ax=axs[i], multiple="fill", kde=False, element="bars", fill=True, stat='density')
                axs[i].set_xlabel(var)

            # Remove any remaining blank subplots
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)


            st.write("Multiclass Barplot")
            # Get the names of all columns with data type 'object' (categorical columns)
            cat_cols = df.columns.tolist()

            # Get the names of all columns with data type 'object' (categorical variables)
            cat_vars = df.select_dtypes(include=['object']).columns.tolist()

            # Exclude 'Country' from the list if it exists in cat_vars
            if target_variable in cat_vars:
                cat_vars.remove(target_variable)

            # Create a figure with subplots, but only include the required number of subplots
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a count plot for each categorical variable
            for i, var in enumerate(cat_vars):
                top_categories = df[var].value_counts().nlargest(10).index
                filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]  # Exclude rows with NaN values in the variable
                sns.countplot(x=var, hue=target_variable, data=filtered_df, ax=axs[i])
                axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

            # Remove any remaining blank subplots
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)


            st.write("Multiclass Histplot")
            # Get the names of all columns with data type 'object' (categorical columns)
            cat_cols = df.columns.tolist()

            # Get the names of all columns with data type 'int'
            int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
            int_vars = [col for col in num_vars if col != target_variable]

            # Create a figure with subplots
            num_cols = len(int_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a histogram for each integer variable with hue='Attrition'
            for i, var in enumerate(int_vars):
                top_categories = df[var].value_counts().nlargest(10).index
                filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]
                sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i])
                axs[i].set_title(var)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)

def page7():
    st.title("EDA with Google Gemini")
    
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV file:")

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Show the original DataFrame
        st.write("Original DataFrame:")
        st.dataframe(df)



        st.write("**Countplot Barchart**")

        # Get the names of all columns with data type 'object' (categorical columns) excluding 'Country'
        cat_vars = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() > 1 and df[col].nunique() <= 10]

        # Create a figure with subplots
        num_cols = len(cat_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a countplot for the top 10 values of each categorical variable using Seaborn
        for i, var in enumerate(cat_vars):
            top_values = df[var].value_counts().head(10).index
            filtered_df = df.copy()
            filtered_df[var] = df[var].apply(lambda x: x if x in top_values else 'Other')
            sns.countplot(x=var, data=filtered_df, ax=axs[i])
            axs[i].set_title(var)
            axs[i].tick_params(axis='x', rotation=90)

        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

        # Adjust spacing between subplots
        fig.tight_layout()

        # Show plots using Streamlit
        st.pyplot(fig)
        fig.savefig("plot4.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        import PIL.Image

        img = PIL.Image.open("plot4.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(img)

        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)


        st.write("**Histoplot**")
        # Get the names of all columns with data type 'int' or 'float'
        num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns]

        # Create a figure with subplots
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a histplot for each numeric variable using Seaborn
        for i, var in enumerate(num_vars):
            sns.histplot(df[var], ax=axs[i], kde=True)
            axs[i].set_title(var)
            axs[i].set_xlabel('')

        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

        # Adjust spacing between subplots
        fig.tight_layout()

        # Show plots using Streamlit
        st.pyplot(fig)
        fig.savefig("plot7.png")

        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        img = PIL.Image.open("plot7.png")
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)
        
        # Select target variable
        target_variable = st.selectbox("Select target variable:", df.columns)
        
        # Select columns for analysis
        columns_for_analysis = st.multiselect("Select columns for analysis:", [col for col in df.columns if col != target_variable])

        # Process button
        if st.button("Process"):
            # Select the target variable and columns for analysis from the original DataFrame
            target_variable_data = df[target_variable]
            columns_for_analysis_data = df[columns_for_analysis]

            # Display target variable in a dataframe
            target_variable_df = df[[target_variable]]
            st.write("Target Variable DataFrame:")
            st.dataframe(target_variable_df)
            
            # Display columns for analysis in a dataframe
            columns_for_analysis_df = df[columns_for_analysis]
            st.write("Columns for Analysis DataFrame:")
            st.dataframe(columns_for_analysis_df)
            
            # Concatenate target variable and columns for analysis into a single DataFrame
            df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)
            st.write("Columns for Analysis and Target Variable DataFrame:")
            st.dataframe(df)

            # Drop columns with null values more than 25%
            null_percentage = df.isnull().sum() / len(df)
            columns_to_drop = null_percentage[null_percentage > 0.25].index
            df.drop(columns=columns_to_drop, inplace=True)

            # Fill missing values below 25% with median
            for col in df.columns:
                if df[col].isnull().sum() > 0:  # Check if there are missing values
                    if null_percentage[col] <= 0.25:
                        if df[col].dtype in ['float64', 'int64']:  # Check if missing values are below 25%
                            median_value = df[col].median()  # Calculate median for the column
                            df[col].fillna(median_value, inplace=True)
            
            # Convert object datatype columns to lowercase
            for col in df.columns:
                if df[col].dtype == 'object':  # Check if datatype is object
                    df[col] = df[col].str.lower()  # Convert values to lowercase

            st.write("Cleaned Dataset")
            st.dataframe(df)


            st.write("**Multiclass Barplot**")
            # Get the names of all columns with data type 'object' (categorical columns)
            cat_cols = df.columns.tolist()

            # Get the names of all columns with data type 'object' (categorical variables)
            cat_vars = df.select_dtypes(include=['object']).columns.tolist()

            # Exclude 'Country' from the list if it exists in cat_vars
            if target_variable in cat_vars:
                cat_vars.remove(target_variable)

            # Create a figure with subplots, but only include the required number of subplots
            num_cols = len(cat_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a count plot for each categorical variable
            for i, var in enumerate(cat_vars):
                top_categories = df[var].value_counts().nlargest(10).index
                filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]  # Exclude rows with NaN values in the variable
                sns.countplot(x=var, hue=target_variable, data=filtered_df, ax=axs[i])
                axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)

            # Remove any remaining blank subplots
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)
            fig.savefig("plot2.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

            import PIL.Image

            img = PIL.Image.open("plot2.png")
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content(img)

            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.write("**Google Gemini Response About Data**")
            st.write(response.text)


            st.write("**Multiclass Histplot**")
            # Get the names of all columns with data type 'object' (categorical columns)
            cat_cols = df.columns.tolist()

            # Get the names of all columns with data type 'int'
            int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
            int_vars = [col for col in int_vars if col != target_variable]

            # Create a figure with subplots
            num_cols = len(int_vars)
            num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
            fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
            axs = axs.flatten()

            # Create a histogram for each integer variable with hue='Attrition'
            for i, var in enumerate(int_vars):
                top_categories = df[var].value_counts().nlargest(10).index
                filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]
                sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i])
                axs[i].set_title(var)

            # Remove any extra empty subplots if needed
            if num_cols < len(axs):
                for i in range(num_cols, len(axs)):
                    fig.delaxes(axs[i])

            # Adjust spacing between subplots
            fig.tight_layout()

            # Show plot
            st.pyplot(fig)
            fig.savefig("plot3.png")

            def to_markdown(text):
                text = text.replace('•', '  *')
                return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

            genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

            import PIL.Image

            img = PIL.Image.open("plot3.png")
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content(img)

            response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image", img], stream=True)
            response.resolve()
            st.write("**Google Gemini Response About Data**")
            st.write(response.text)


            # Chat feature with Google Gemini
            st.write("Ask questions about the visualizations:")
            user_question = st.text_input("")

            if user_question:
                plot_names = ["plot", "plot2", "plot3", "plot4"]

                for plot_name in plot_names:
                    if plot_name in user_question.lower():
                        try:
                            img = PIL.Image.open(f"{plot_name}.png")
                            model = genai.GenerativeModel('gemini-pro-vision')
                            response = model.generate_content(["You are a professional Data Analyst, answer the question: " + user_question, img], stream=True)
                            response.resolve()
                            st.write(response.text)
                        except Exception as e:
                            print(f"Error: {e}")
                            st.write("An error occurred while using Gemini. Please try again later.")
                        break

                else:
                    try:
                        img = PIL.Image.open("combined_plots.png")  # Replace with combined image if needed
                        model = genai.GenerativeModel('gemini-pro-vision')
                        response = model.generate_content(["You are a professional Data Analyst, answer the question: " + user_question, img], stream=True)
                        response.resolve()
                        st.write(response.text)
                    except Exception as e:
                        print(f"Error: {e}")
                        st.write("An error occurred while using Gemini. Please try again later.")




def page8():
    import io
    import textwrap
    import json
    st.title("Summarize your CSV File")
    
    # Upload the CSV file
    uploaded_file = st.file_uploader("Upload CSV file:")

    # Check if the file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Show the original DataFrame
        st.write("Original DataFrame:")
        st.dataframe(df)
        df.dropna(inplace=True)

        # Convert the DataFrame to a string variable
        df_string = df.to_string()

        # Configure genai with API key
        genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")

        # Instantiate the model
        model = genai.GenerativeModel('gemini-1.0-pro-latest')

        # Generate content
        response = model.generate_content(["You are a Professional Data Analyst, Make a Summary and actionable insight based on the csv dataset here :", df_string], stream=True)
        response.resolve()
        st.write("**Google Gemini Response About Data**")
        st.write(response.text)
        


# Use a session state variable to keep track of the selected page
session_state = st.session_state
if 'selected_page' not in session_state:
    session_state.selected_page = "Classification Prediction"

st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #222021;
            color: #f1f2f6;
            width: 100%;
            transition: background-color 0.3s, box-shadow 0.3s;
        }
        div.stButton > button:hover {
            background-color: #222021;
            color: #ff5c5d;  /* Adjust the text color on hover */
            width: 100%;
            box-shadow: 0 0 5px 2px #ff5c5d, 0 0 10px 5px #ff5c5d, 0 0 15px 7.5px #ff5c5d;
        }
        [data-testid="stSidebar"] {
            background: #222021;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation with styled buttons
page_1_button = st.sidebar.button("Classification Prediction", key="page1", type="primary")
page_2_button = st.sidebar.button("Regression Prediction", key="page2", type="primary")
page_3_button = st.sidebar.button("NLP : PDF Document Analysis", key="page3", type="primary")
page_4_button = st.sidebar.button("Sentiment Analysis", key="page4", type="primary")
page_5_button = st.sidebar.button("Clustering", key="page5", type="primary")
page_6_button = st.sidebar.button("Exploratory Data Analysis", key="page6", type="primary")
page_7_button = st.sidebar.button("EDA with Google Gemini", key="page7", type="primary")
page_8_button = st.sidebar.button("Summarize your CSV File", key="page8", type="primary")

# Display selected page based on the button clicked
if page_1_button:
    session_state.selected_page = "Classification Prediction"

if page_2_button:
    session_state.selected_page = "Regression Prediction"

if page_3_button:
    session_state.selected_page = "NLP : PDF Document Analysis"

if page_4_button:
    session_state.selected_page = "Sentiment Analysis"

if page_5_button:
    session_state.selected_page = "Clustering"

if page_6_button:
    session_state.selected_page = "Exploratory Data Analysis"

if page_7_button:
    session_state.selected_page = "EDA with Google Gemini"

if page_8_button:
    session_state.selected_page = "Summarize your CSV File"

# Inject the button style into the app
button_style = """
    <style>
        .css-1vru0uf {
            border: none !important;
            color: white;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            background-color: transparent !important;
        }
        .css-1vru0uf:hover {
            background-color: red;
        }
    </style>
"""

# Display selected page
if session_state.selected_page == "Classification Prediction":
    page1()
elif session_state.selected_page == "Regression Prediction":
    page2()
elif session_state.selected_page == "NLP : PDF Document Analysis":
    page3()
elif session_state.selected_page == "Sentiment Analysis":
    page4()
elif session_state.selected_page == "Clustering":
    page5()
elif session_state.selected_page == "Exploratory Data Analysis":
    page6()
elif session_state.selected_page == "EDA with Google Gemini":
    page7()
elif session_state.selected_page == "Summarize your CSV File":
    page8()
# Inject the button style into the app
st.markdown(button_style, unsafe_allow_html=True)



