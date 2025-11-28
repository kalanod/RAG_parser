from pathlib import Path
from typing import List

from core.services import (
    DocumentIngestionService,
    QAService,
    VectorStoreService,
)
from utils import prompts


class RagContext:
    name: str
    files: List[List]

    def __init__(self, name, embedder, llm, db_dir="./db"):
        self.name = name
        self.files = []
        self.db_dir = db_dir
        self.embedder = embedder
        self.agent_prompt_template = prompts.agent_prompt_template

        self.document_ingestion_service = DocumentIngestionService()
        self.vector_store_service = VectorStoreService(
            collection_name=self.name,
            embedder=self.embedder,
            persist_directory=self.db_dir,
        )
        self.qa_service = QAService(
            llm_model_name=llm,
            vector_store_service=self.vector_store_service,
            agent_prompt_template=self.agent_prompt_template,
        )

    def add_file(self, path: Path):
        documents = self.document_ingestion_service.parse_and_clean(path)
        self.vector_store_service.add_documents(documents)

    def edit_name(self, new_name: str):
        self.name = new_name

    def embedd_files(self, documents):
        return self.embedder.embed_documents([i.page_content for i in documents])

    def embedd_query(self, query):
        return self.embedder.embed_query(query)

    def search_in_chroma(self, query, top_k: int = 5):
        return self.vector_store_service.search(query, top_k)

    def new_question(self, question: str):
        return self.qa_service.answer_question(question)

    def agent_question(self, question):
        return self.qa_service.agent_answer(question)

    def test(self):
        print(self.qa_service.llm)
