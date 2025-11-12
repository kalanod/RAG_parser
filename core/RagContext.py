from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.pipeline import parse_document, load_from_chroma, save_to_chroma


class RagContext:
    name: str
    files: List[List]
    db = None
    db_dir = None

    def __init__(self, name, embedder, db_dir="../db"):
        self.name = name
        self.files = []
        self.db_dir = db_dir
        self.embedder = embedder

    def add_file(self, path: Path):
        file = self.embedd_file(path)
        self.files.append(file)

    def edit_name(self, new_name: str):
        self.name = new_name

    def embedd_files(self):
        for i, file in enumerate(self.files):
            if file[1] is None:
                self.files[i] = self.embedd_file(file[0])

    def embedd_file(self, path):
        documents = parse_document(path)
        embeddings = self.embedder.embed_documents([i.page_content for i in documents])
        return [path, documents, embeddings]

    def embedd_query(self, query):
        return self.embedder.embed_query(query)

    def save_to_db(self):
        save_to_chroma(
            persist_directory=self.db_dir,
            documents=[doc[1] for doc in self.files],
            embeddings=[doc[2] for doc in self.files],
            collection_name=self.name,
        )

    def get_db(self):
        if self.db is None:
            self.db = load_from_chroma(self.db_dir, self.name)
        return self.db

    def rerank_context(self, docs):
        return docs[0][0]

    def generate_answer(self, question, context):
        return context

    def normalize_question(self, question):
        return question

    def search_in_chroma(self, query_vector, top_k: int = 5):
        results = self.db._collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )
            distance = results["distances"][0][i]
            docs.append((doc, distance))
        return docs

    def new_question(self, question: str, context=""):
        question = self.normalize_question(question)
        query_embedding = self.embedd_query(question)
        results = self.search_in_chroma(query_embedding, top_k=3)
        context = context + "\n" + self.rerank_context(results)
        answer = self.generate_answer(question, context)
        return answer
