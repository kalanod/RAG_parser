from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.pipeline import parse_document, create_embeddings, init_chroma_with_embeddings, load_from_chroma, \
    search_in_chroma


class RagContext:
    name: str
    files: List[List]
    db = None
    db_dir = None

    def __init__(self, name, base_url, model, api_key, db_dir="../db"):
        self.name = name
        self.files = []
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.db_dir = db_dir

    def add_file(self, file: Path):
        self.files.append([file, None, None])
        self.embedd_files()
        return self

    def edit_name(self, new_name: str):
        self.name = new_name
        return self

    def embedd_files(self):
        for i, file in enumerate(self.files):
            if file[1] is None:
                documents = parse_document(file[0])
                embeddings = create_embeddings(documents, self.base_url, self.model, self.api_key)
                self.files[i] = [file[0], documents, embeddings]

    def save_to_db(self):
        init_chroma_with_embeddings(
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

    def new_question(self, question: str, context=""):
        question = self.normalize_question(question)
        query_embedding = create_embeddings(
            [Document(page_content=question)],
            self.base_url,
            self.model,
            self.api_key
        )
        results = search_in_chroma(self.get_db(), query_embedding, top_k=3)
        context = context + "\n" + self.rerank_context(results)
        answer = self.generate_answer(question, context)
        return answer
