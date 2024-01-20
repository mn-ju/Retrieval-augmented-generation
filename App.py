
from src.utils import collection_dropdown, file_uploader,llm_model
import streamlit as st
from pathlib import Path
import pandas as pd
from src.logger import create_logger



def App():
    st.set_page_config(page_title='Information Retrieval WebApp')
    # Header
    st.title('Information Retrieval WebApp')
    st.markdown('---')

    # Create a centered container for layout
    container = st.container()
    list_of_collections = collection_dropdown()
    # Add a "None" option to the list
    list_of_collections = ['None'] + list_of_collections
    selected_option = st.selectbox("Select an option", list_of_collections, index=0)  # Set the default index to 0 (which is "None")
    st.write("You selected:", selected_option)
    collection_name = selected_option

    # Check if selected_option is "None"
    if selected_option == "None":
        uploaded_file = st.file_uploader('Upload an article', type='txt', key='file_upload')
        if uploaded_file is not None:
            file_name = uploaded_file.name
            collection_name = Path(uploaded_file.name).stem
            Embeddings = file_uploader(uploaded_file,collection_name)

                
    st.header('Query')
    result = []
    with st.form('myform'):
        query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')

        # Submit button
        submitted = st.form_submit_button('Submit', disabled=not (query_text))
        if submitted:
            with st.spinner('Loading...'):
                response = llm_model(query_text, collection_name)  
                result.append(response)

    # Results Section
    st.header('Results')
    if len(result) > 0:
        for idx, response in enumerate(result, start=1):
            st.subheader(f'Result {idx}')
            st.write(response)
    else:
        st.warning('No results found. Please adjust your query or upload an article.')

    # Footer
    st.markdown('---')
    st.markdown('Created with Large Language Model')
   

    
App()