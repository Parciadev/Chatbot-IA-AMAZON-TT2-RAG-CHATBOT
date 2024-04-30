import streamlit as st 
import utemia_backend as demo 

st.set_page_config(page_title="Prototipo-UtemIA-Chatbot") 

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Reglamento estudiantil RAG - Q & A Prototipo-UtemIA-Chatbot 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True) 

if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Generando prototipo...."): 
        st.session_state.vector_index = demo.utemia_index() 

input_text = st.text_area("Input text", label_visibility="collapsed") 
go_button = st.button("Ingresar respuesta", type="primary") 

if go_button: 
    
    with st.spinner("📢 Generando respuesta...."): 
        response_content = demo.utemia_rag_response(index=st.session_state.vector_index, question=input_text) ### replace with RAG Function from backend file
        st.write(response_content) 