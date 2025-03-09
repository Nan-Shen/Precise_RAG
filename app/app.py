import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('../keys/.env')
load_dotenv(dotenv_path=dotenv_path)

from util import FullChain

if 'reference' not in st.session_state:
    print('Searching for reference doc.....')
    fp = "../data"
    file_path = None
    for f in os.listdir(fp):
        if f.endswith('.pdf'):
           file_path = '/'.join([fp, f])
    if file_path == None:
        print('Reference not found. Continue in NO RAG mode.')
        chain = FullChain(None).build()
        st.session_state["local_model"] = chain
        print('Complete loading instruct model(without RAG).')
    else:
        print(f'Found reference file {file_path}. Continue in RAG mode.')
        rag_chain = FullChain(file_path).build()
        st.session_state["local_model"] = rag_chain
        st.session_state["reference"] = file_path
        print('Compelete initializing RAG chain!')

def rag_ans(messages, cache=False):
    """
    """
    question = messages[-1]['content']
    print(question)
    if cache:
        history_ans = cache_answers(messages, question)
        if history_ans is None:
            pass
        else:
            return history_ans  
    model = st.session_state["local_model"]
    response = model.invoke(question)
    print(response)
    return response

def cache_answers(messages, question):
    for i, history_message in enumerate(messages):
        if history_message['content'] == question:
            j = i
            while j < len(messages):
                if history_message['role'] != 'assistant':
                    j += 1
                else:
                    answer = history_message['content']
                    return answer
    return None

def main():    
    print('start')
    title = ('<p style="font-size: 20px;">Precise RAG Test with NVIDIA 10K 2025</p>')
    st.markdown(title, unsafe_allow_html=True)

    message = st.chat_message("assistant")
    if 'reference' in st.session_state:
        message.write('Hello! I can answer your questions based on NVIDIA 10K published in 2025.')
    else:
        message.write('Hello! I can answer your questions. But I have no reference in my library right now.')
        
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        print('one')

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        print('two')

    # React to user input
    if prompt := st.chat_input("Type in a question anout NVIDIA. More companies will be avail soon."):
        print('three')
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        responses = rag_ans(messages)
        
        message_placeholder.markdown(responses)
        print('four')
        st.session_state.messages.append({"role": "assistant", "content": responses})

if __name__ == "__main__":
    main()
 
    
