import streamlit as st
import openai

def run_chat_ui():
    st.subheader("🤖 Ask the Data Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("💬 Your question about this app or your data")

    if st.button("Ask"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages,
                    temperature=0.3
                )
                assistant_reply = response.choices[0].message["content"]
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                st.markdown(assistant_reply)
            except Exception as e:
                st.error(f"Chat failed: {e}")
