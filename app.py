import streamlit as st
import pandas as pd
import json
import uuid  # For generating unique IDs
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from chunking import RecursiveTokenChunker, LLMAgenticChunker, ProtonxSemanticChunker
from utils import process_batch, divide_dataframe, get_search_result

import time
import pdfplumber  # PDF extraction
import io
from docx import Document  # DOCX extraction
from components import notify
from constant import NO_CHUNKING, GEMINI, EN, VI, USER, ASSISTANT, ENGLISH, VIETNAMESE

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]


# Initialize the page
st.title("Drag and Drop RAG")



# Initialize session state for language choice and model embedding
if "language" not in st.session_state:
    st.session_state.language = EN  # Default language is English
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None  # Placeholder for the embedding model

# Language selection popup
st.sidebar.subheader("Choose Language")
language_choice = st.sidebar.radio("Select language:", [ENGLISH, VIETNAMESE])

# Switch embedding model based on language choice
if language_choice == ENGLISH:
    if st.session_state.language and st.session_state.language != EN:
        st.session_state.language = EN
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        st.sidebar.success("Using English embedding model: all-MiniLM-L6-v2")
    else:
        st.session_state.language = EN
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        st.sidebar.success("Using English embedding model: all-MiniLM-L6-v2")
elif language_choice == VIETNAMESE:
    if st.session_state.language and st.session_state.language != VI:
        st.session_state.language = VI
        st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        st.sidebar.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")

# Sidebar settings
st.sidebar.header("Settings")

# Chunk size input
st.session_state.chunk_size = st.sidebar.number_input(
    "Chunk Size", min_value=50, max_value=1000, value=200, step=50, help="Set the size of each chunk in terms of tokens."
)

st.session_state.number_docs_retrieval = st.sidebar.number_input(
    "Number of documnents retrieval", min_value=1, max_value=50, value=10, step=1, help="Set the number of document which will be retrieved."
)


if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None

# Initialize session state for chroma client, collection, and model
if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")

# Initialize session state for collection and model
if "collection" not in st.session_state:
    st.session_state.collection = None

# Check if the collection exists, if not, create a new one
if st.session_state.collection is None:
    st.session_state.random_collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
    st.session_state.collection = st.session_state.client.get_or_create_collection(
        name=st.session_state.random_collection_name,
        metadata={"description": "A collection for RAG system"},
    )


# Step 1: File Upload (CSV, JSON, PDF, or DOCX) and Column Detection
uploaded_files = st.file_uploader(
    "Upload CSV, JSON, PDF, or DOCX files", 
    type=["csv", "json", "pdf", "docx"], 
    accept_multiple_files=True
)

# Initialize a variable for tracking the success of saving the data
st.session_state.data_saved_success = False

# Ensure `df` is only accessed if it's been created and has columns
if uploaded_files is not None:
    all_data = []
    
    for uploaded_file in uploaded_files:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            all_data.append(df)

        elif uploaded_file.name.endswith(".json"):
            json_data = json.load(uploaded_file)
            df = pd.json_normalize(json_data)  # Normalize JSON to a flat DataFrame format
            all_data.append(df)

        elif uploaded_file.name.endswith(".pdf"):
            # Extract text from PDF
            pdf_text = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    pdf_text.append(page.extract_text())

            # Convert PDF text into a DataFrame (assuming one column for simplicity)
            df = pd.DataFrame({"content": pdf_text})
            all_data.append(df)

        elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
            # Extract text from DOCX
            doc = Document(io.BytesIO(uploaded_file.read()))
            docx_text = [para.text for para in doc.paragraphs if para.text]

            # Convert DOCX text into a DataFrame (assuming one column for simplicity)
            df = pd.DataFrame({"content": docx_text})
            all_data.append(df)

    # Concatenate all data into a single DataFrame
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        st.dataframe(df)

        doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]

        if "doc_ids" not in st.session_state:
            st.session_state.doc_ids = doc_ids

        # Add or replace the '_id' column in the DataFrame
        df['doc_id'] = st.session_state.doc_ids

        st.subheader("Chunking")

        # **Ensure `df` is not empty before calling selectbox**
        if not df.empty:
            # Display selectbox to choose the column for vector search
            index_column = st.selectbox("Choose the column to index (for vector search):", df.columns)
            st.write(f"Selected column for indexing: {index_column}")
        else:
            st.warning("The DataFrame is empty, please upload valid data.")
            
        # Disable the "AgenticChunker" option if the API key is not provided
        chunk_options = [
            NO_CHUNKING,
            "RecursiveTokenChunker", 
            "SemanticChunker",
            "AgenticChunker",
        ]

        # Step 4: Chunking options
        if not st.session_state.get("gemini_api_key") and st.session_state.get("chunkOption") == "AgenticChunker":
            currentChunkerIdx = 0
            st.session_state.chunkOption = NO_CHUNKING
            notify("You have to setup the GEMINI API KEY FIRST in the Setup LLM Section", "error")
        elif not st.session_state.get("chunkOption"):
            currentChunkerIdx = 0
            st.session_state.chunkOption = NO_CHUNKING
        else:
            currentChunkerIdx = chunk_options.index(st.session_state.get("chunkOption")) 
        
        st.radio(
            "Please select one of the options below.",
            chunk_options,
            captions=[
                "Keep the original document",
                "Recursively chunks text into smaller, meaningful token groups based on specific rules or criteria.",
                "Chunking with semantic comparison between chunks",
                "Let LLM decide chunking (requires Gemini API)"
            ],
            key="chunkOption",
            index=currentChunkerIdx
        )
        
        chunkOption = st.session_state.get("chunkOption")
        
        if chunkOption == "SemanticChunker":
            embedding_option = st.selectbox(
                "Choose the embedding method for Semantic Chunker:",
                ["TF-IDF", "Sentence-Transformers"]
            )
        chunk_records = []

        # Iterate over rows in the original DataFrame
        for index, row in df.iterrows():

            # For "No Chunking" option, treat the selected index column as a single "chunk"
            chunker = None
            selected_column_value = row[index_column]
            chunks = []
            if not(type(selected_column_value) == str and len(selected_column_value) > 0):
                continue
            if chunkOption == NO_CHUNKING:
                # Use the selected index_column
                chunks = [selected_column_value]
                
            # For "RecursiveTokenChunker" option, split text from the selected index column into smaller chunks
            elif chunkOption == "RecursiveTokenChunker":
                chunker = RecursiveTokenChunker(
                    chunk_size=200
                )
                chunks = chunker.split_text(selected_column_value)
                
            elif chunkOption == "SemanticChunker":
                if embedding_option == "TF-IDF":
                    chunker = ProtonxSemanticChunker(
                        embedding_type="tfidf",
                    )
                else:
                    chunker = ProtonxSemanticChunker(
                        embedding_type="transformers", 
                        model="all-MiniLM-L6-v2",
                    )
                chunks = chunker.split_text(selected_column_value)
            elif chunkOption == "AgenticChunker" and  st.session_state.get("gemini_api_key"):
                chunker = LLMAgenticChunker(
                    organisation="google", 
                    model_name="gemini-1.5-pro", 
                    api_key=st.session_state.get('gemini_api_key')
                )
                chunks = chunker.split_text(selected_column_value)
            # For each chunk, add a dictionary with the chunk and original_id to the list
            for chunk in chunks:
                chunk_record = {**row.to_dict(), 'chunk': chunk}
                
                # Rearrange the dictionary to ensure 'chunk' and '_id' come first
                chunk_record = {
                    'chunk': chunk_record['chunk'],
                    # '_id': str(uuid.uuid4()),
                    **{k: v for k, v in chunk_record.items() if k not in ['chunk', '_id']}
                }
                chunk_records.append(chunk_record)

        # Convert the list of dictionaries to a DataFrame
        chunks_df = pd.DataFrame(chunk_records)

        # Display the result
        st.write("Number of chunks:", len(chunks_df))
        st.dataframe(chunks_df)



# Button to save data
# Button to save data
if st.button("Save Data"):
    try:
        collection = st.session_state.collection

        # Define the batch size
        batch_size = 256

        # Split the DataFrame into smaller batches
        df_batches = divide_dataframe(chunks_df, batch_size)

        # Check if the dataframe has data, otherwise show a warning and skip the processing
        if not df_batches:
            st.warning("No data available to process.")
        else:
            num_batches = len(df_batches)

            # Initialize progress bar
            progress_text = "Saving data to Chroma. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            # Process each batch
            for i, batch_df in enumerate(df_batches):
                if batch_df.empty:
                    continue  # Skip empty batches (just in case)

                process_batch(batch_df, st.session_state.embedding_model, collection)

                # Update progress dynamically for each batch
                progress_percentage = int(((i + 1) / num_batches) * 100)
                my_bar.progress(progress_percentage, text=f"Processing batch {i + 1}/{num_batches}")

                time.sleep(0.1)  # Optional sleep to simulate processing time

            # Empty the progress bar once completed
            my_bar.empty()

            st.success("Data saved to Chroma vector store successfully!")
            st.markdown("Collection name: `{}`".format(st.session_state.random_collection_name))
            st.session_state.data_saved_success = True

    except Exception as e:
        st.error(f"Error saving data to Chroma: {str(e)}")


# Show blue tick if data has been saved successfully
header_i = 1
header_text = "{}. Setup data ✅".format(header_i) if st.session_state.data_saved_success else "{}. Setup data".format(header_i)
st.header(header_text)



if st.session_state.data_saved_success:
    st.markdown("✅ **Data Saved Successfully!**")



# Step 3: Define which columns LLMs should answer from
columns_to_answer = None
if uploaded_files:
    columns_to_answer = st.multiselect(
        "Select one or more columns LLMs should answer from (multiple selections allowed):", 
        df.columns
    )

# Step 2: Setup LLMs (Gemini Only)
header_i += 1
header_text_llm = "{}. Setup LLMs ✅".format(header_i) if 'gemini_model' in st.session_state else "{}. Setup LLMs".format(header_i)
st.header(header_text_llm)
# Example user selection
llm_choice = st.selectbox("Choose Model Source:", ["Online",])

if llm_choice == "Online":
    # Initialize a variable for tracking if the API key was entered successfully
    api_key_success = False
    st.session_state.llm_type = GEMINI
    # Input Gemini API key
    st.markdown("Obtain the API key from the [Google AI Studio](https://ai.google.dev/aistudio/).")
    st.text_input(
        "Enter your Gemini API Key:", 
        type="password", 
        key="gemini_api_key")   
    
    if st.session_state.get('gemini_api_key'):
        genai.configure(api_key=st.session_state.get('gemini_api_key'))
        st.success("Gemini API Key saved successfully!")
        st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        api_key_success = True

    # Show blue tick if API key was entered successfully
    if api_key_success:
        st.markdown("✅ **API Key Saved Successfully!**")
# Step 3: Interactive Chatbot
header_i += 1
st.header("{}. Interactive Chatbot".format(header_i))

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# URL of the Flask API

# Display the chat history using chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": USER, "content": prompt})
    # Display user message in chat message container
    with st.chat_message(USER):
        st.markdown(prompt)
    # Display assistant response in chat message container
    # Prepare the payload for the request

    with st.chat_message(ASSISTANT):
        if st.session_state.collection is not None:
            # Combine retrieved data to enhance the prompt based on selected columns
            if columns_to_answer:
                metadatas, retrieved_data = get_search_result(
                    st.session_state.embedding_model, 
                    prompt, 
                    st.session_state.collection, 
                    columns_to_answer,
                    st.session_state.number_docs_retrieval
                )

               # Flatten the list of lists (metadatas) and convert to a DataFrame
                if metadatas:
                    flattened_metadatas = [item for sublist in metadatas for item in sublist]  # Flatten the list of lists
                    
                    # Convert the flattened list of dictionaries to a DataFrame
                    metadata_df = pd.DataFrame(flattened_metadatas)
                    
                    # Display the DataFrame in the sidebar
                    st.sidebar.subheader("Retrieval data")
                    st.sidebar.dataframe(metadata_df)
                else:
                    st.sidebar.write("No metadata to display.")
                
                enhanced_prompt = """You are a good salesperson. The prompt of the customer is: "{}". Answer it based on the following retrieved data: \n{}""".format(prompt, retrieved_data)

                if st.session_state.llm_type == GEMINI:
                    # Step 3: Feed enhanced prompt to Gemini LLM for completion
                    response = st.session_state.gemini_model.generate_content(enhanced_prompt)

                    content = response.candidates[0].content.parts[0].text

                    # Display the extracted content in the Streamlit app
                    st.markdown(content)

                    # Update chat history
                    st.session_state.chat_history.append({"role": ASSISTANT, "content": content})
                elif st.session_state.llm_type == "local":
                    messages = [
                        {
                            "content": enhanced_prompt,
                            "role": USER
                        }
                    ]

                    response = st.session_state.local_llms.chat(messages)

                    st.markdown(response['content'])
                    st.session_state.chat_history.append(response['content'])
            else:
                st.warning("Please select columns for the chatbot to answer from.")
        else:
            st.error("No collection found. Please upload data and save it first.")

