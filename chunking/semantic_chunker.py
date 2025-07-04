from .base_chunker import BaseChunker
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
print(nltk.data.path)
print(nltk.__file__)


class ProtonxSemanticChunker(BaseChunker):
    def __init__(self, threshold=0.3, embedding_type="tfidf", model="all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.embedding_type = embedding_type
        self.model = model

        # Download punkt for sentence tokenization, ensuring it's only done when class is initialized
        nltk.download("punkt", quiet=True)

    def embed_function(self, sentences):
        """
        Embeds sentences using the specified embedding method.
        Supports 'tfidf' and 'transformers' embeddings.
        """
        # Bỏ các câu rỗng hoặc toàn khoảng trắng
        sentences = [s.strip() for s in sentences if s and s.strip()]

        if not sentences:
            raise ValueError("Input sentences are empty or only contain whitespace.")

        if self.embedding_type == "tfidf":
            vectorizer = TfidfVectorizer()
            try:
                return vectorizer.fit_transform(sentences).toarray()
            except ValueError as e:
                raise ValueError(f"TF-IDF vectorization failed: {e}")

        elif self.embedding_type == "transformers":
            self.model = SentenceTransformer(self.model)
            return self.model.encode(sentences)

        else:
            raise ValueError("Unsupported embedding type. Choose 'tfidf' or 'transformers'.")

    def split_text(self, text):
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s and s.strip()]

        if not sentences:
            return []

        try:
            vectors = self.embed_function(sentences)
        except ValueError as e:
            print("Embedding error:", e)
            return []  # Trả về rỗng nếu embedding lỗi

        similarities = cosine_similarity(vectors)
        chunks = [[sentences[0]]]

        for i in range(1, len(sentences)):
            sim_score = similarities[i - 1, i]
            if sim_score >= self.threshold:
                chunks[-1].append(sentences[i])
            else:
                chunks.append([sentences[i]])

        return [' '.join(chunk) for chunk in chunks]

        