# Import các thư viện cần thiết
import streamlit as st  # Framework để tạo web application
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
import json  # Xử lý định dạng JSON
import uuid  # Tạo ID ngẫu nhiên duy nhất
import time  # Xử lý thời gian
import io  # Xử lý input/output
from typing import List, Optional, Dict, Any  # Type hints cho Python

# Import các thư viện specialized
from sentence_transformers import SentenceTransformer  # Chuyển văn bản thành vector
import chromadb  # Vector database để lưu trữ và tìm kiếm
import google.generativeai as genai  # API của Google Gemini
import pdfplumber  # Đọc file PDF
from docx import Document  # Đọc file Word

# Import các module tự định nghĩa
from chunking import RecursiveTokenChunker, ProtonxSemanticChunker
from utils import process_batch, divide_dataframe, get_search_result
from components import notify
from constant import (
    NO_CHUNKING, GEMINI, EN, VI, USER, ASSISTANT,
    ENGLISH, VIETNAMESE
)


def clear_session_state() -> None:
    """
    Xóa toàn bộ session state.
    Hữu ích khi cần reset ứng dụng về trạng thái ban đầu.
    """
    for key in st.session_state.keys():
        del st.session_state[key]


def initialize_session_state() -> None:
    """
    Khởi tạo các biến session state cần thiết cho ứng dụng.
    Bao gồm ngôn ngữ, models, collection, và lịch sử chat.
    """
    # Khởi tạo ngôn ngữ mặc định là tiếng Anh
    if "language" not in st.session_state:
        st.session_state.language = EN

    # Khởi tạo các model là None
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = None

    # Khởi tạo ChromaDB client và collection
    if "client" not in st.session_state:
        st.session_state.client = chromadb.PersistentClient("db")
    if "collection" not in st.session_state:
        st.session_state.collection = None
        # Tạo tên collection ngẫu nhiên
        st.session_state.random_collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
        st.session_state.collection = st.session_state.client.get_or_create_collection(
            name=st.session_state.random_collection_name,
            metadata={"description": "A collection for RAG system"},
        )

    # Khởi tạo lịch sử chat và trạng thái lưu dữ liệu
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "data_saved_success" not in st.session_state:
        st.session_state.data_saved_success = False
    if "llm_type" not in st.session_state:
        st.session_state.llm_type = GEMINI
    if "chunkOption" not in st.session_state:
        st.session_state.chunkOption = NO_CHUNKING


def setup_language_model(language_choice: str) -> None:
    """
    Cấu hình model ngôn ngữ dựa trên lựa chọn của người dùng.

    Args:
        language_choice (str): Ngôn ngữ được chọn (ENGLISH hoặc VIETNAMESE)
    """
    try:
        if language_choice == ENGLISH:
            if st.session_state.language and st.session_state.language != EN:
                # Xóa collection cũ khi đổi ngôn ngữ
                if st.session_state.collection:
                    try:
                        st.session_state.client.delete_collection(st.session_state.random_collection_name)
                    except:
                        pass
                st.session_state.language = EN
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Tạo collection mới
                st.session_state.random_collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
                st.session_state.collection = st.session_state.client.get_or_create_collection(
                    name=st.session_state.random_collection_name,
                    metadata={"description": "A collection for RAG system"},
                )
                st.sidebar.success("Using English embedding model: all-MiniLM-L6-v2")
            else:
                st.session_state.language = EN
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.sidebar.success("Using English embedding model: all-MiniLM-L6-v2")
        elif language_choice == VIETNAMESE:
            try:
                if st.session_state.language and st.session_state.language != VI:
                    # Xóa collection cũ khi đổi ngôn ngữ
                    if st.session_state.collection:
                        try:
                            st.session_state.client.delete_collection(st.session_state.random_collection_name)
                        except:
                            pass
                    st.session_state.language = VI
                    st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
                    # Tạo collection mới
                    st.session_state.random_collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
                    st.session_state.collection = st.session_state.client.get_or_create_collection(
                        name=st.session_state.random_collection_name,
                        metadata={"description": "A collection for RAG system"},
                    )
                    st.sidebar.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")
            except Exception as e:
                st.error(f"Error loading Vietnamese model: {str(e)}")
                st.session_state.language = EN
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.warning("Falling back to English model")
    except Exception as e:
        st.error(f"Error loading language model: {str(e)}")


def process_uploaded_files(uploaded_files: List) -> Optional[pd.DataFrame]:
    """
    Xử lý các file được upload và trả về DataFrame tổng hợp.

    Args:
        uploaded_files (List): Danh sách các file được upload

    Returns:
        Optional[pd.DataFrame]: DataFrame chứa dữ liệu từ các file, None nếu không có file
    """
    if not uploaded_files:
        return None

    all_data = []
    for uploaded_file in uploaded_files:
        # Xử lý file CSV
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        # Xử lý file JSON    
        elif uploaded_file.name.endswith(".json"):
            json_data = json.load(uploaded_file)
            df = pd.json_normalize(json_data)
        # Xử lý file PDF    
        elif uploaded_file.name.endswith(".pdf"):
            pdf_text = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    pdf_text.append(page.extract_text())
            df = pd.DataFrame({"content": pdf_text})
        # Xử lý file Word    
        elif uploaded_file.name.endswith((".docx", ".doc")):
            doc = Document(io.BytesIO(uploaded_file.read()))
            docx_text = [para.text for para in doc.paragraphs if para.text]
            df = pd.DataFrame({"content": docx_text})
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else None


def process_chunking(df: pd.DataFrame, index_column: str, chunk_option: str) -> pd.DataFrame:
    """
    Chia nhỏ văn bản thành các đoạn dựa trên phương pháp được chọn.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        index_column (str): Tên cột cần chia nhỏ
        chunk_option (str): Phương pháp chia nhỏ

    Returns:
        pd.DataFrame: DataFrame mới chứa các đoạn văn bản đã chia nhỏ
    """
    chunk_records = []

    for _, row in df.iterrows():
        selected_column_value = row[index_column]
        # Bỏ qua nếu giá trị không phải string hoặc rỗng
        if not (isinstance(selected_column_value, str) and selected_column_value):
            continue

        chunks = []
        # Không chia nhỏ
        if chunk_option == NO_CHUNKING:
            chunks = [selected_column_value]
        # Chia nhỏ theo token    
        elif chunk_option == "RecursiveTokenChunker":
            chunker = RecursiveTokenChunker(chunk_size=st.session_state.chunk_size)
            chunks = chunker.split_text(selected_column_value)
        # Chia nhỏ theo ngữ nghĩa    
        elif chunk_option == "SemanticChunker":
            chunker = ProtonxSemanticChunker(embedding_type="tfidf")
            chunks = chunker.split_text(selected_column_value)

        # Tạo bản ghi cho mỗi chunk
        for chunk in chunks:
            chunk_record = {**row.to_dict(), 'chunk': chunk}
            chunk_records.append(chunk_record)

    return pd.DataFrame(chunk_records)


def handle_chat(prompt: str, collection, columns_to_answer: List[str], number_docs: int) -> None:
    """
    Xử lý tương tác chat với người dùng.

    Args:
        prompt (str): Câu hỏi của người dùng
        collection: ChromaDB collection chứa dữ liệu
        columns_to_answer (List[str]): Danh sách cột được sử dụng để trả lời
        number_docs (int): Số lượng document cần lấy ra để trả lời
    """
    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Thêm câu hỏi mới vào lịch sử
    st.session_state.chat_history.append({"role": USER, "content": prompt})
    with st.chat_message(USER):
        st.markdown(prompt)

    # Xử lý câu trả lời
    with st.chat_message(ASSISTANT):
        # Kiểm tra điều kiện
        if collection is None:
            st.error("No collection found. Please upload data and save it first.")
            return

        if not columns_to_answer:
            st.warning("Please select columns for the chatbot to answer from.")
            return

        if not st.session_state.data_saved_success:
            st.error("Please save your data first before chatting!")
            return

        # Tìm kiếm thông tin liên quan
        metadata, retrieved_data = get_search_result(
            st.session_state.embedding_model,
            prompt,
            collection,
            columns_to_answer,
            number_docs
        )

        # Hiển thị metadata nếu có
        if metadata:
            flattened_metadatas = [item for sublist in metadata for item in sublist]
            metadata_df = pd.DataFrame(flattened_metadatas)
            st.sidebar.subheader("Retrieval data")
            st.sidebar.dataframe(metadata_df)

        # Tạo prompt nâng cao và gửi đến Gemini
        enhanced_prompt = f"""You are a good salesperson. The prompt of the customer is: "{prompt}". 
        Answer it based on the following retrieved data: \n{retrieved_data}"""

        if st.session_state.llm_type == GEMINI:
            try:
                response = st.session_state.gemini_model.generate_content(enhanced_prompt)
                content = response.candidates[0].content.parts[0].text
                st.markdown(content)
                st.session_state.chat_history.append({"role": ASSISTANT, "content": content})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")


def main():
    """
    Hàm chính của ứng dụng.
    Thiết lập giao diện và xử lý tương tác người dùng.
    """
    # Thiết lập tiêu đề và khởi tạo
    st.title("Drag and Drop RAG")
    initialize_session_state()

    # Lựa chọn ngôn ngữ
    st.sidebar.subheader("Choose Language")
    language_choice = st.sidebar.radio("Select language:", [ENGLISH, VIETNAMESE])
    setup_language_model(language_choice)

    # Cài đặt các thông số
    st.sidebar.header("Settings")
    st.session_state.chunk_size = st.sidebar.number_input(
        "Chunk Size", min_value=50, max_value=1000, value=200, step=50,
        help="Set the size of each chunk in terms of tokens."
    )

    st.session_state.number_docs_retrieval = st.sidebar.number_input(
        "Number of documents retrieval", min_value=1, max_value=50, value=10, step=1,
        help="Set the number of document which will be retrieved."
    )

    # Cài đặt Gemini API
    st.header("1. Setup LLM")
    st.markdown("Obtain the API key from the [Google AI Studio](https://ai.google.dev/aistudio/).")
    api_key = st.text_input("Enter your Gemini API Key:", type="password", key="gemini_api_key")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            st.success("✅ Gemini API Key configured successfully!")
        except Exception as e:
            st.error(f"Error configuring Gemini API: {str(e)}")

    # Phần upload và xử lý file
    st.header("2. Setup Data" + (" ✅" if st.session_state.data_saved_success else ""))
    uploaded_files = st.file_uploader(
        "Upload CSV, JSON, PDF, or DOCX files",
        type=["csv", "json", "pdf", "docx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload some files to get started!")
        return

    # Xử lý dữ liệu
    columns_to_answer = []
    df = process_uploaded_files(uploaded_files)

    if df is not None:
        st.dataframe(df)

        # Tạo document ID cho mỗi hàng trong DataFrame
        doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]
        # Lưu document ID vào session state
        if "doc_ids" not in st.session_state:
            st.session_state.doc_ids = doc_ids
        df['doc_id'] = doc_ids

        # Cho phép người dùng chọn cột để LLM trả lời từ
        columns_to_answer = st.multiselect(
            "Select one or more columns LLMs should answer from (multiple selections allowed):",
            df.columns
        )

        # Phần cấu hình và xử lý chunking
        st.subheader("Chunking")
        if not df.empty:
            # Cho phép người dùng chọn cột để index
            index_column = st.selectbox(
                "Choose the column to index (for vector search):",
                df.columns
            )

            # Các tùy chọn chunking
            chunk_options = [
                NO_CHUNKING,  # Giữ nguyên văn bản
                "RecursiveTokenChunker",  # Chia theo số token
                "SemanticChunker",  # Chia theo ngữ nghĩa
            ]

            # Radio button để chọn phương pháp chunking
            st.radio(
                "Please select one of the options below.",
                chunk_options,
                captions=[
                    "Keep the original document",
                    "Recursively chunks text into smaller, meaningful token groups",
                    "Chunking with semantic comparison between chunks",
                ],
                key="chunkOption",
                index=chunk_options.index(st.session_state.chunkOption)
            )

            # Thực hiện chunking và hiển thị kết quả
            chunks_df = process_chunking(df, index_column, st.session_state.chunkOption)
            st.write("Number of chunks:", len(chunks_df))
            st.dataframe(chunks_df)

            # Button để lưu dữ liệu vào ChromaDB
            if st.button("Save Data"):
                if not st.session_state.embedding_model:
                    st.error("Please select a language first to initialize the embedding model!")
                    return

                try:
                    # Lấy collection từ session state
                    collection = st.session_state.collection
                    # Xử lý theo batch để tránh quá tải
                    batch_size = 256
                    df_batches = divide_dataframe(chunks_df, batch_size)

                    if df_batches:
                        # Hiển thị progress bar
                        progress_text = "Saving data to Chroma. Please wait..."
                        my_bar = st.progress(0, text=progress_text)

                        # Xử lý từng batch
                        for i, batch_df in enumerate(df_batches):
                            if not batch_df.empty:
                                # Xử lý batch và cập nhật progress
                                process_batch(batch_df, st.session_state.embedding_model, collection)
                                progress = int(((i + 1) / len(df_batches)) * 100)
                                my_bar.progress(progress, text=f"Processing batch {i + 1}/{len(df_batches)}")

                        # Xóa progress bar và hiển thị thông báo thành công
                        my_bar.empty()
                        st.success("Data saved to Chroma vector store successfully!")
                        st.session_state.data_saved_success = True
                        st.markdown("Collection name: `{}`".format(st.session_state.random_collection_name))

                except Exception as e:
                    st.error(f"Error saving data to Chroma: {str(e)}")

        # Giao diện chat
    st.header("3. Interactive Chat")
    # Chat input
    if prompt := st.chat_input("What is up?"):
        # Kiểm tra Gemini API đã được cấu hình chưa
        if not st.session_state.gemini_model:
            st.error("Please setup Gemini API key first!")
            return

        # Xử lý chat
        handle_chat(
            prompt,
            st.session_state.collection,
            columns_to_answer,
            st.session_state.number_docs_retrieval
        )


if __name__ == "__main__":
    main()