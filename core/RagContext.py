import hashlib
import os
from pathlib import Path
from typing import List

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from core.pipeline import parse_document, load_from_chroma
from utils import prompts, get_logger

LOGGER = get_logger(__name__)
class RagContext:
    name: str
    files: List[List]
    db = None
    db_dir = None
    agent_prompt_template = prompts.agent_prompt_template

    def __init__(self, name, embedder, llm, db_dir="./db"):
        self.name = name
        self.files = []
        self.db_dir = db_dir
        self.embedder = embedder
        os.makedirs(self.db_dir, exist_ok=True)
        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=llm,
            max_new_tokens=300,
            temperature=0.1,
            repetition_penalty=1.1
        ))
        self.db = Chroma(
            collection_name=self.name,
            persist_directory=self.db_dir,
            embedding_function=self.embedder,
        )
        self.id_cache = set(self.db.get(include=[]).get("ids", []))

        self.memory = ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            return_messages=True
        )
        self.tools = [self.summarize, self.search_tool]
        self.agent = create_react_agent(self.llm, self.tools, self.agent_prompt_template)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            return_full_text=True
        )

    @tool(description="get additional context to provided query")
    def search_tool(self, query, top_k: int = 5):
        results = self.search_in_chroma(query, top_k)
        results = self.rerank_results(results)
        return results

    @tool(description="summarizing context entire file")
    def summarize(self):
        return "результат"

    def get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_file(self, path: Path):
        documents = parse_document(path)
        self.save_to_db(documents)

    def edit_name(self, new_name: str):
        self.name = new_name

    def embedd_files(self, documents):
        return self.embedder.embed_documents([i.page_content for i in documents])

    def embedd_query(self, query):
        return self.embedder.embed_query(query)

    def save_to_db(self, documents):
        new_docs = []
        new_ids = []

        for doc in documents:
            doc_id = self.get_hash(doc.page_content)
            if doc_id in self.id_cache:
                continue
            self.id_cache.add(doc_id)
            new_docs.append(doc)
            new_ids.append(doc_id)

        if new_docs:
            self.db.add_documents(new_docs, ids=new_ids)

    def get_db(self):
        if self.db is None:
            self.db = load_from_chroma(self.db_dir, self.name, self.embedder)
        return self.db

    def rerank_results(self, docs):
        return docs[0].page_content + "\n" + docs[1].page_content

    def generate_answer(self, question, context):
        prompt = f"""
        дай ответ на чёткий и точный ответ на вопрос, используй для ответа строго выделенный контекст, пиши только ответ, не добавляй никаких своих комментариев, кратко и по делу. Если информации нет в контексте скажи "к сожалению информация об этом не найдена"

        Context:
        {context}

        Question:
        {question}

        Answer:
            """
        output = self.llm.invoke(prompt).split("Answer:")[1]
        return output

    def normalize_question(self, question):
        LOGGER.info(f"context: {self.name} - normalizing question: {question}")
        prompt = prompts.question_normalization_prompt
        output = self.llm.invoke(prompt.from_template({"input": question}))
        LOGGER.info(f"context: {self.name} - normalized question: {output}")
        return output

    def search_in_chroma(self, query, top_k: int = 5):
        LOGGER.info(f"context: {self.name} - search in chroma: {query}")
        results = self.db.similarity_search(query, k=top_k)
        LOGGER.info(f"context: {self.name} - search in chroma results: {results}")
        return results

    def new_question(self, question: str):
        LOGGER.info(f"context: {self.name} - new question: {question}")
        question = self.normalize_question(question)
        results = self.search_in_chroma(question, top_k=3)
        context = self.rerank_results(results)
        answer = self.generate_answer(question, context)
        return answer

    def agent_question(self, question):
        LOGGER.info(f"context: {self.name} - new question: {question}")
        question = self.normalize_question(question)
        answer = self.agent_executor.invoke({"input": question})["output"]
        return answer

    def test(self):
        print(self.llm)
