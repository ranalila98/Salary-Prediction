import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_RemoteWork = data["le_RemoteWork"]

def show_predict_page():
    st.header("Software Developer Salary Prediction")
    st.write("""### Please provide the following information to predict the salary""")
    countries = (
        "United States of America",
        "India",
        "United Kingdom of Great Britain and Northern Ireland",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Norway",
        "Sweden",
    )
    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    Work = ("Onsite","Remote", "Hybrid")

    country = st.selectbox("Country", countries, format_func=lambda x: 'Select Country' if x == '' else x)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)
    workingstyle = st.selectbox("Working Style", Work)

    # country = st.selectbox("Country", countries)
    # education = st.selectbox("Education Level", education)
    # expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience, workingstyle]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X[:, 3] = le_RemoteWork.transform(X[:,3])
        X = X.astype(float)
        X = pd.DataFrame(X, columns=['Country', 'EdLevel', 'YearsCodePro', 'RemoteWork'])
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")