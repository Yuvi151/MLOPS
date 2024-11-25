import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the App
st.title("CSV Data Visualization App")

# Sidebar for file upload
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    # Select visualization type
    st.sidebar.header("Visualization Options")
    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        ["Line Chart", "Bar Chart", "Histogram"]
    )

    # Select columns for visualization
    if chart_type != "Histogram":
        x_axis = st.sidebar.selectbox("Select X-Axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-Axis", df.columns)
    else:
        column = st.sidebar.selectbox("Select Column for Histogram", df.columns)

    # Plotting based on user selection
    if chart_type == "Line Chart":
        st.write("### Line Chart")
        st.line_chart(df[[x_axis, y_axis]].set_index(x_axis))
    elif chart_type == "Bar Chart":
        st.write("### Bar Chart")
        st.bar_chart(df[[x_axis, y_axis]].set_index(x_axis))
    elif chart_type == "Histogram":
        st.write("### Histogram")
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=20, color='skyblue', edgecolor='black')
        st.pyplot(fig)

else:
    st.write("Please upload a CSV file to start.")
