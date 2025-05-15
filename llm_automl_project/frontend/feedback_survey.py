import streamlit as st

st.subheader("User Feedback Survey")

useful = st.slider("How useful was this platform?", 1, 5)
easy = st.slider("How easy was it to use?", 1, 5)
trust = st.slider("How much do you trust the output?", 1, 5)

if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
    # Save to file or database if needed
