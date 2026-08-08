import streamlit as st
import pandas as pd

st.title("OUTFIT STYLE RECOMMENDER")
st.write("Welcome to the OUTFIT STYLE RECOMMENDER!")

df = pd.read_csv(r"clothing_data.csv")

st.sidebar.header("User Preferences")
gender=st.sidebar.selectbox("Select Gender",df['gender'].unique())
occasion=st.sidebar.selectbox("Select Occasion",df['occasion'].unique())
weather=st.sidebar.selectbox("Select Weather",df['weather'].unique())
color=st.sidebar.selectbox("Select Color",df['color'].unique())


filtered = df[(df['gender'] == gender) & 
              (df['occasion'] == occasion) & 
              (df['weather'] == weather) & 
              (df['color'] == color)
              ]

if len(filtered)==0:
    st.warning("No outfits found for the selected preferences. Please try different options.")
    filtered = df[(df['gender'] == gender) & 
                  (df['occasion'] == occasion)
    ]

if st.button("Show Recommendation"):
    st.subheader("Recommended Outfits")
    for i in range(min(3,len(filtered))):
        style = filtered.iloc[i]['style']
        st.write(f"**{style.title()}**")
        
    st.info("Note: The recommendations are based on the selected preferences. You can adjust your preferences in the sidebar to see different outfit suggestions.")