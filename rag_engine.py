import fitz
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from vision_pipeline import VisionPipeline


class MultimodalRAG:
    def __init__(self, groq_key: str):
        self.vision = VisionPipeline(groq_key)
        self.embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def process_pdf(self, pdf_bytes: bytes) -> List[Document]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        documents = []

        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                for chunk in splitter.split_text(text):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"page": i + 1, "modality": "text"}
                    ))

            for img in page.get_images():
                image = doc.extract_image(img[0])["image"]
                fact = self.vision.analyze_image(image, i + 1)

                if not fact or fact.confidence == "low":
                    continue

                documents.append(Document(
                    page_content=(
                        f"On page {fact.page}, {fact.description}. "
                        f"Trend: {fact.trend}. "
                        f"X-axis: {fact.x_label}. Y-axis: {fact.y_label}. "
                        f"Data points: {fact.data_points}."
                    ),
                    metadata={
                        "page": fact.page,
                        "modality": "vision",
                        "image_type": fact.image_type,
                        "confidence": fact.confidence
                    }
                ))

        return documents

    def build_store(self, docs: List[Document]) -> FAISS:
        if not docs:
            raise RuntimeError("No valid content extracted from PDF")

        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        return FAISS.from_texts(texts, self.embedder, metadatas=metas)

    def retrieve(self, store, query: str, modality=None):
        docs = store.similarity_search(query, k=12)
        return [
            d for d in docs
            if (not modality or d.metadata.get("modality") == modality)
        ]
