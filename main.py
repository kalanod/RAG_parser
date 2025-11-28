import os
from pathlib import Path

import pytesseract
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from core.RagContext import RagContext
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')
EMBEDDER = HuggingFaceEmbeddings(
    model_name='distiluse-base-multilingual-cased-v2',
    model_kwargs={'device': 'cpu', "trust_remote_code": True},
    encode_kwargs={'normalize_embeddings': True},
)
llm_name = "Qwen2.5-1.5B-Instruct"

if __name__ == '__main__':
    context = RagContext("test", EMBEDDER, llm_name)
    context.add_file(Path("./res/Программа визита ГПН-Шельф в ТМК НТЦ.docx"))
    # print("во сколько будет выступать Нургалеев А.Р?\n" + context.new_question("во сколько будет Кофе-брейк?"))
    # print()
    print("а кто контактное лицо на мероприятии?\n" + context.new_question("а кто контактное лицо на мероприятии?"))
    #context.agent_question("во сколько будет Кофе-брейк?")
    context.test()
