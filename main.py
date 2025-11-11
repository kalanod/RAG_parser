import os
from pathlib import Path
from dotenv import load_dotenv

from core.RagContext import RagContext

load_dotenv()
LLM_KEY = os.getenv("LLM_KEY")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


if __name__ == '__main__':
    context = RagContext("main", EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_KEY)
    context.add_file(Path("./res/diapi.pdf"))
    context.new_question("сколько будут платить?")
