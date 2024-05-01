import streamlit as st
import utemia_backend as demo

st.set_page_config(page_title="Prototipo-UtemIA-Chatbot")

logo_filename = "UTEM.png"
st.image(logo_filename, width=300)

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Prototipo-UtemIA-Chatbot 🎯 RAG - Q & A </p>'
st.markdown(new_title, unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner("📀 Generando prototipo...."):
        st.session_state.vector_index = demo.utemia_index()

input_text = st.text_area("Chatea con UtemIA aquí...", label_visibility="collapsed")
go_button = st.button("Enviar", key="response_button", help="Presiona para recibir una respuesta del chatbot")

if go_button:
    with st.spinner("📢 Generando respuesta...."):
        response_content = demo.utemia_rag_response(index=st.session_state.vector_index, question=input_text)
        st.write(response_content)