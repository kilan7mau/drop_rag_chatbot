# Drop RAG chatbot

## Overview


**Drag and Drop RAG** là một hệ thống **Retrieval-Augmented Generation (RAG)** cho phép người dùng tải lên dữ liệu (CSV, JSON, PDF hoặc DOCX), lưu trữ vào **Chroma vector store** và trò chuyện với chatbot sử dụng **Gemini** (`gemini-1.5-pro`). Chatbot sẽ truy xuất các thông tin liên quan từ file bạn tải lên, tăng cường truy vấn, và trả kết quả sử dụng **Large Language Models (LLMs)**.

## Tính năng

1. **Tải lên nhiều định dạng**: Hỗ trợ CSV, JSON, PDF, DOCX; cho phép chọn cột để tìm kiếm vector.
2. **Lưu trữ & truy xuất embedding**: Sử dụng **Chroma**.
3. **Chatbot tương tác**: Dựa trên **Gemini API** để trả lời dựa vào dữ liệu người dùng tải lên.
4. **Tuỳ chỉnh LLM**: Chọn cột dữ liệu mà LLM sẽ dùng để trả lời.

## Cài đặt & Chạy ứng dụng

1. Clone repository:
   ```bash
   git clone https://github.com/kilan7mau/drop_rag_chatbot.git
   cd drop_rag_chatbot
   ```

2. Cài đặt các thư viện Python cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

3. Chạy ứng dụng Streamlit:
   ```bash
   streamlit run app.py
   ```

4. Truy cập ứng dụng tại địa chỉ [http://localhost:8501](http://localhost:8501) trên trình duyệt.

## Hướng dẫn sử dụng

1. **Upload Data**: Tải lên file CSV, JSON, PDF hoặc DOCX. Chọn cột để lập chỉ mục vector.
2. **Save Data**: File được lưu vào Chroma vector store với vector embedding sinh bởi model `all-MiniLM-L6-v2` hoặc `keepitreal/vietnamese-sbert`.
3. **Setup LLMs**: Nhập **Gemini API key** để cấu hình chatbot. Lấy key tại [Google AI Studio](https://aistudio.google.com/app/apikey).
4. **Chat**: Bắt đầu trò chuyện với chatbot, hệ thống sẽ truy xuất và tăng cường kết quả dựa trên dữ liệu bạn đã tải.

## Lưu ý

- Bắt buộc phải có **Gemini API key** để sử dụng chatbot.
- Đảm bảo dữ liệu đã lưu vào **Chroma vector store** trước khi chat.

## Xử lý sự cố

- Nếu không truy xuất được dữ liệu, hãy kiểm tra lại cột dữ liệu đã chọn.
- Đảm bảo API key hợp lệ và bộ sưu tập (collection) đã được khởi tạo.

---

**Liên hệ đóng góp:**  
Nếu bạn muốn đóng góp hoặc có thắc mắc, hãy tạo Issue hoặc Pull Request trên GitHub.

---