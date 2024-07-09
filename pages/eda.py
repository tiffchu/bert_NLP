import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("EDA Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataframe:")
    st.write(df)

    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns for analysis", columns)

    if selected_columns:
        st.write("Selected Columns:")
        st.write(df[selected_columns].describe())

        plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot", "Scatter Matrix", "Correlation Heatmap"])

        if plot_type == "Histogram":
            for column in selected_columns:
                st.write(f"Histogram for {column}")
                fig, ax = plt.subplots()
                df[column].hist(ax=ax, bins=30)
                st.pyplot(fig)

        elif plot_type == "Boxplot":
            for column in selected_columns:
                st.write(f"Boxplot for {column}")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column], ax=ax)
                st.pyplot(fig)

        elif plot_type == "Scatter Matrix":
            st.write("Scatter Matrix")
            fig = sns.pairplot(df[selected_columns])
            st.pyplot(fig)

        elif plot_type == "Correlation Heatmap":
            st.write("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    
    date_column = st.selectbox("Select date column for time series analysis (if any)", ["None"] + columns)

    if date_column != "None":
        df[date_column] = pd.to_datetime(df[date_column])
        time_series_column = st.selectbox("Select column for time series analysis", [col for col in columns if col != date_column])

        if time_series_column:
            st.write(f"Time Series Analysis for {time_series_column}")
            fig, ax = plt.subplots()
            df.set_index(date_column)[time_series_column].plot(ax=ax)
            st.pyplot(fig)

