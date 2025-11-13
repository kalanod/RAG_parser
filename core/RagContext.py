from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma

from core.pipeline import parse_document, load_from_chroma


class RagContext:
    name: str
    files: List[List]
    db = None
    db_dir = None

    def __init__(self, name, embedder, db_dir="./db"):
        self.name = name
        self.files = []
        self.db_dir = db_dir
        self.embedder = embedder
        self.db = Chroma(
            collection_name=self.name,
            persist_directory=self.db_dir,
            embedding_function=self.embedder,
        )


    def add_file(self, path: Path):
        documents = parse_document(path)
        self.save_to_db(documents)

    def edit_name(self, new_name: str):
        self.name = new_name

    def embedd_file(self, documents):
        return self.embedder.embed_documents([i.page_content for i in documents])

    def embedd_query(self, query):
        return self.embedder.embed_query(query)

    def save_to_db(self, documents, embeddings=None):
        if embeddings is not None:
            self.db.add_documents(documents=documents, embeddings=embeddings)
        else:
            self.db.add_documents(documents=documents)

    def get_db(self):
        if self.db is None:
            self.db = load_from_chroma(self.db_dir, self.name, self.embedder)
        return self.db

    def rerank_context(self, docs):
        return docs[0][0]

    def generate_answer(self, question, context):
        return context

    def normalize_question(self, question):
        return question

    def search_in_chroma(self, query, top_k: int = 5):
        results = self.db.similarity_search(query, k=top_k)
        return results

    def new_question(self, question: str, context=""):
        question = self.normalize_question(question)
        results = self.search_in_chroma(question, top_k=3)
        context = context + "\n" + self.rerank_context(results)
        answer = self.generate_answer(question, context)
        return answer
