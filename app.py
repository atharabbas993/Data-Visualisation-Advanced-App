# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from ydata_profiling import ProfileReport
import tempfile
import os

# Set the title of the Streamlit app
st.title("Advanced Data Visualization App")

# Add a file uploader to allow users to upload their dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the dataset into a pandas DataFrame
    data_frame = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset
    st.write("### Preview of the Dataset")
    st.dataframe(data_frame.head())

    # Allow users to view random samples of the dataset
    if st.checkbox("Show Random Samples"):
        sample_size = st.slider("Select sample size", 1, 100, 10)
        st.write(f"### Random {sample_size} Samples from the Dataset")
        st.dataframe(data_frame.sample(sample_size))

    # Display basic dataset information
    st.write("### Dataset Information")
    st.write(f"Number of Rows: {data_frame.shape[0]}")
    st.write(f"Number of Columns: {data_frame.shape[1]}")
    st.write("Column Names:")
    st.write(data_frame.columns.tolist())

    # Generate the Pandas Profiling report
    st.write("### Pandas Profiling Report")
    if st.button("Generate Pandas Profiling Report"):
        try:
            # Use minimal mode for faster report generation
            prof = ProfileReport(data_frame, minimal=True)
            st.write("Report generated successfully!")

            # Save the report to a temporary HTML file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                prof.to_file(output_file=tmpfile.name)
                tmpfile_path = tmpfile.name

            # Display the report in the Streamlit app
            with open(tmpfile_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, width=1000, height=1500, scrolling=True)

            # Add a download button for the report
            with open(tmpfile_path, "rb") as f:
                report_bytes = f.read()
            st.download_button(
                label="Download Pandas Profiling Report",
                data=report_bytes,
                file_name="pandas_profiling_report.html",
                mime="text/html"
            )

            # Clean up the temporary file
            os.unlink(tmpfile_path)
        except Exception as e:
            st.error(f"Error generating report: {e}")

    # Column Selection for Visualization
    st.write("### Column Selection for Visualization")
    selected_columns = st.multiselect("Select columns to visualize", data_frame.columns)

    if selected_columns:
        st.write(f"#### Visualizations for Selected Columns: {', '.join(selected_columns)}")

        for column in selected_columns:
            st.write(f"##### Column: {column}")

            # Check the data type of the column
            if data_frame[column].dtype == "object" or data_frame[column].nunique() < 10:  # Categorical or low-cardinality column
                st.write(f"**Pie Chart for {column}**")
                fig = px.pie(data_frame, names=column, title=f"Pie Chart for {column}")
                st.plotly_chart(fig)

                st.write(f"**Bar Plot for {column}**")
                fig = px.bar(data_frame[column].value_counts(), title=f"Bar Plot for {column}")
                st.plotly_chart(fig)

            elif pd.api.types.is_numeric_dtype(data_frame[column]):  # Numeric column
                st.write(f"**Histogram for {column}**")
                fig = px.histogram(data_frame, x=column, title=f"Histogram for {column}")
                st.plotly_chart(fig)

                st.write(f"**Boxplot for {column}**")
                fig = px.box(data_frame, y=column, title=f"Boxplot for {column}")
                st.plotly_chart(fig)

                st.write(f"**Violin Plot for {column}**")
                fig = px.violin(data_frame, y=column, title=f"Violin Plot for {column}")
                st.plotly_chart(fig)

            elif pd.api.types.is_datetime64_any_dtype(data_frame[column]):  # Datetime column
                st.write(f"**Time Series Plot for {column}**")
                fig = px.line(data_frame, x=column, y=data_frame.select_dtypes(include=["int64", "float64"]).columns[0], title=f"Time Series Plot for {column}")
                st.plotly_chart(fig)

            else:
                st.write(f"**Unsupported data type for {column}**")

    # Correlation Heatmap for Numeric Columns
    st.write("### Correlation Heatmap")
    numeric_columns = data_frame.select_dtypes(include=["int64", "float64"]).columns
    if not numeric_columns.empty and len(numeric_columns) >= 2:  # Check if numeric_columns is not empty
        # Increase figure size and DPI for better readability
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # Create the heatmap with larger annotation font size
        sns.heatmap(
            data_frame[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",  # Format annotations to 2 decimal places
            annot_kws={"size": 12},  # Increase annotation font size
            ax=ax
        )
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.yticks(rotation=0)   # Keep y-axis labels horizontal
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for a correlation heatmap.")

    # Pairplot for Numeric Columns
    st.write("### Pairplot for Numeric Columns")
    if not numeric_columns.empty and len(numeric_columns) >= 2:  # Check if numeric_columns is not empty
        # Convert numeric_columns to a list for st.multiselect
        numeric_columns_list = numeric_columns.tolist()
        default_columns = numeric_columns_list[:min(3, len(numeric_columns_list))]  # Ensure default is not out of range
        selected_pairplot_columns = st.multiselect("Select columns for pairplot", numeric_columns_list, default=default_columns)
        if selected_pairplot_columns:
            fig = sns.pairplot(data_frame[selected_pairplot_columns])
            st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for a pairplot.")

else:
    st.write("Please upload a CSV file to get started.")