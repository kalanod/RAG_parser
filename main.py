import pytesseract
from langchain_huggingface import HuggingFaceEmbeddings
from core.RagContext import RagContext

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
EMBEDDER = HuggingFaceEmbeddings(
    model_name='distiluse-base-multilingual-cased-v2',
    model_kwargs={'device': 'cpu', "trust_remote_code": True},
    encode_kwargs={'normalize_embeddings': True},
)
llm_name = "Qwen2.5-1.5B-Instruct"

if __name__ == '__main__':
    context = RagContext("dop", EMBEDDER, llm_name)
    # context.add_file(Path("./res/Программа визита ГПН-Шельф в ТМК НТЦ.docx"))
    print(context.new_question("во сколько будет выступать Тюльдин?"))
