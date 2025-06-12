import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    page_title="Student Habits vs Performance",
    layout="centered",
    page_icon="🍏",
)

df = pd.read_csv("student_habits_performance.csv")
st.sidebar.title("Student Habits vs Student Performance")
page = st.sidebar.selectbox("Select Page",["Introduction","Data Visualization", "Automated Report","Predictions"])

if page == "Introduction":
    st.title("Student Performance Predictor")
    st.subheader("Analyzing the effects of student habits on academic performance")
    st.markdown("""
    #### What this app does:
    - *Analyzes* key lifestyle, academic, and personal factors
    - *Visualizes* trends and provides actionable insights
    - *Predicts* student academic performance (exam sores) using a linear regression model
    """)
    st.image("dataset-card.png", width=500)

    st.markdown("#### The Dataset")
    st.markdown("""
    This dataset was collected to study the relationship between *student lifestyle and academic habits on performance outcomes (exam scores)*. It includes variables such as:
    - 📚 Class attendance and time studying
    - 🧠 Mental health
    - 💤 Sleep habits
    - 🍎 Diet
    - 🏃‍♂️ Exercise/physical activity 
    - 👩‍💻 Social media and netflix usage 
                
    The goal is to leverage these factors to *predict the academic success* in students and better understand
    which variables are most impactful.
    """)

    missing = df.isnull().sum()
    if missing.sum() != 0:
        df = df.dropna()
    
    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("#####  Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Data Visualization":
    st.subheader("Data Viz")

    df_numeric = df.select_dtypes(include=np.number)

    tab1, tab2, tab3 = st.tabs(["Scatter Plot","Box Plot", "Correlation Heatmap"])
    with tab1:
        st.subheader("Scatter Plot")
        fig_bar, ax_bar = plt.subplots(figsize=(12,6))
        x_col = st.selectbox("Select x-axis variable", df_numeric.columns.drop(["exam_score","age", "exercise_frequency", "mental_health_rating"]))
        sns.scatterplot(df, x = x_col, y = "exam_score")
        st.pyplot(fig_bar)
    with tab2:
        st.subheader("Box Plot")
        fig_bar, ax_bar = plt.subplots(figsize=(12,6))
        x_col_2 = st.selectbox("Select x-axis variable", options=["exercise_frequency", "diet_quality", "mental_health_rating"])
        sns.boxplot(df, x = x_col_2, y = "exam_score")
        st.pyplot(fig_bar)
    with tab3:
        st.subheader("Correlation Matrix")

        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        
        st.pyplot(fig_corr)

elif page == "Automated Report":
    st.subheader("Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="Student Habits vs Student Performance",explorative=True,minimal=True)
            st_profile_report(profile)
        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="student_habits_performance.html",mime='text/html')

elif page == "Predictions":
    st.subheader("Predictions")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    list_non_num =["student_id","gender","part_time_job","diet_quality","parental_education_level","internet_quality","extracurricular_participation"]
    for element in list_non_num:
        df[element]= le.fit_transform(df[element])
    
    list_var = list(df.columns.drop(["student_id"]))
    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    target_selection  = st.sidebar.selectbox("Select target variable (Y))",list_var)
    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    X = df[features_selection]
    y = df[target_selection]

    st.dataframe(X.head())
    st.dataframe(y.head())

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    from sklearn import metrics 
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")


    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)






