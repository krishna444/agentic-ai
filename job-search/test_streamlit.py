import streamlit as st
import random
import pandas as pd

with st.form("my_form"):
    name=st.text_input("Name")
    surname=st.text_input("Surname")
    age=st.slider('How old are you',0,122)
    start_day=st.date_input("Starting date")
    
    submitted=st.form_submit_button("Submit")
    if submitted:
        st.write(f'Name  = {name} start_day={start_day}')
st.write("Outside of form object")