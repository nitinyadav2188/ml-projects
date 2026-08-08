import streamlit as st
import pickle
import numpy as np
import plotly.express as px

model = pickle.load(open('code.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

st.title("Health Risk Predictor")
age = st.slider("Age",18,80)
diet = st.selectbox("Diet Quality", ['Poor', 'Average', 'Good'])
exercise = st.slider("Exercise Day per week",0,7,3)
sleep = st.slider("Sleep Hours",2,12,6)
stress = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
bmi = st.number_input("BMI", 10.0, 40.0, 22.0)
smoking = st.selectbox("Smoking Status", ['Yes', 'No'])
alcohol = st.selectbox("Alcohol Consumption", ['Low', 'Medium', 'High'])
family_history = st.selectbox("Family History of Disease", ['Yes', 'No'])

if st.button("Predict Risk"):
    input_data = np.array([[age, 
                            encoders['diet'].transform([diet])[0], 
                            exercise, 
                            sleep, 
                            encoders['stress'].transform([stress])[0], 
                            bmi, 
                            encoders['smoking'].transform([smoking])[0], 
                            encoders['alcohol'].transform([alcohol])[0], 
                            encoders['family_history'].transform([family_history])[0]]])

    prediction = model.predict(input_data)
    probs = model.predict(input_data)

    risk_label = encoders['risk_level'].inverse_transform([prediction[0]])[0]

    st.success(risk_label)

    # Bar Chart for lifestyle factors
    factors = {
        "Diet": encoders['diet'].transform([diet])[0],
        "Exercise": exercise,
        "Sleep": sleep,
        "stress": encoders['stress'].transform([stress])[0],
        "BMI": bmi
    }

    bar_fig = px.bar(
        x=list(factors.keys()), 
        y=list(factors.values()), 
        labels={'x':'Lifestyle Factors', 'y':'Values'}, 
        title="Yours Lifestyle Factors"
        )

    st.plotly_chart(bar_fig)