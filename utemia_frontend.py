import streamlit as st
import requests
import uuid


# Function to initialize session ID
def init_session_id():
    session_id = uuid.uuid4()
    st.session_state.session_id = str(session_id)


# Initialize session ID if not already initialized
if 'session_id' not in st.session_state:
    init_session_id()


# Function for generating LLM response
def generate_response(prompt_input):
    response = requests.request("POST", response_url, headers=llm_headers, data=prompt_input)
    print(response.text)
    if response.status_code == 200:
        resp_json = response.json()
        st.sidebar.json(resp_json)
        send_resp = str(resp_json['Response']).strip()
        return send_resp
    else:
        print(response.text)
        return "Ups, algo salio mal, intentalo mas tarde"


# Check if 'messages' is already in st.session_state, if not initialize it
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola, soy Utemia, tu asistente de AWS. ¿En qué puedo ayudarte hoy?"}]

st.set_page_config(page_title="UTEMIA - AWS RAG Chatbot ")
st.sidebar.title("Diseñado Por Pablo Garcia / German Ramirez")
st.sidebar.info(st.session_state.session_id)
st.sidebar.warning("NOTA: Actualizar la pagina olvidara la actual conversacion!")

st.image ("UTEM.png", use_column_width = True)

response_url="https://mcrue9r5vi.execute-api.us-east-1.amazonaws.com/Prod/response" # URL de la API Lambda en la instancia EC2

llm_headers = {
    'Content-Type': 'text/plain',
    'session_id': st.session_state.session_id
}


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# User-provided prompt
if user_prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt, unsafe_allow_html=True)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_prompt)
                st.markdown(response, unsafe_allow_html=True)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)