# The below frontend code is provided by AWS and Streamlit. I have only modified it to make it look attractive.
import streamlit as st 
import utemia_backend as demo ### replace rag_backend with your backend filename

st.set_page_config(page_title="Reglamento estudiantil Q & A Prototipo-UtemIA-Chatbot") ### Modify Heading

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">HR Q & A Prototipo-Utemsito-Chatbot 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True) ### Modify Title

if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Generando Vector DB...."): ###spinner message
        st.session_state.vector_index = demo.utemia_index() ### Your Index Function name from Backend File

input_text = st.text_area("Input text", label_visibility="collapsed") 
go_button = st.button("Ingresar respuesta", type="primary") ### Button Name

if go_button: 
    
    with st.spinner("📢 Generando respuesta...."): ### Spinner message
        response_content = demo.utemia_rag_response(index=st.session_state.vector_index, question=input_text) ### replace with RAG Function from backend file
        st.write(response_content) 