from typing import List

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

from utils import prompts, get_logger
from core.services.vector_store import VectorStoreService


class QAService:
    """Coordinate LLM calls and retrieval operations."""

    def __init__(
        self,
        llm_model_name: str,
        vector_store_service: VectorStoreService,
        agent_prompt_template=prompts.agent_prompt_template,
    ):
        self.logger = get_logger(__name__)
        self.vector_store_service = vector_store_service
        self.agent_prompt_template = agent_prompt_template
        self.llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=llm_model_name,
                max_new_tokens=300,
                temperature=0.1,
                repetition_penalty=1.1,
            )
        )

        self.memory = ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            return_messages=True,
        )
        self.tools = [self.summarize, self.search_tool]
        self.agent = create_react_agent(self.llm, self.tools, self.agent_prompt_template)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            return_full_text=True,
        )

    @tool(description="get additional context to provided query")
    def search_tool(self, query: str, top_k: int = 5):
        results = self.vector_store_service.search(query, top_k)
        results = self.rerank_results(results)
        return results

    @tool(description="summarizing context entire file")
    def summarize(self):
        return "результат"

    def rerank_results(self, docs: List):
        return docs[0].page_content + "\n" + docs[1].page_content

    def generate_answer(self, question: str, context: str):
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

    def normalize_question(self, question: str):
        return question
        self.logger.info(
            "collection: %s - normalizing question: %s",
            self.vector_store_service.collection_name,
            question,
        )
        prompt = prompts.question_normalization_prompt
        output = self.llm.invoke(prompt.format_messages(input=question))
        self.logger.info(
            "collection: %s - normalized question: %s",
            self.vector_store_service.collection_name,
            output,
        )
        return output

    def answer_question(self, question: str):
        self.logger.info(
            "collection: %s - new question: %s",
            self.vector_store_service.collection_name,
            question,
        )
        normalized_question = self.normalize_question(question)
        results = self.vector_store_service.search(normalized_question, top_k=3)
        context = self.rerank_results(results)
        answer = self.generate_answer(normalized_question, context)
        return answer

    def agent_answer(self, question: str):
        self.logger.info(
            "collection: %s - new question: %s",
            self.vector_store_service.collection_name,
            question,
        )
        normalized_question = self.normalize_question(question)
        answer = self.agent_executor.invoke({"input": normalized_question})["output"]
        return answer

    def todo_future_improvements(self):
        """Placeholder for future QA orchestration features."""
        # TODO: add reranker integration, streaming responses, and guardrails.
        return None