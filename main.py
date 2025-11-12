from pathlib import Path
import pytesseract
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.RagContext import RagContext

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MAIN_EMBEDDER = HuggingFaceEmbeddings(
    model_name='ai-sage/Giga-Embeddings-instruct',
    model_kwargs={'device': 'cpu', "trust_remote_code": True},
    encode_kwargs={'normalize_embeddings': True},
)

if __name__ == '__main__':
    context = RagContext("main", MAIN_EMBEDDER)
    context.add_file(Path("./res/1.pdf"))
    context.new_question("сколько будут платить?")
