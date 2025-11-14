import pytesseract
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.RagContext import RagContext

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
EMBEDDER = HuggingFaceEmbeddings(
    model_name='distiluse-base-multilingual-cased-v2',
    model_kwargs={'device': 'cpu', "trust_remote_code": True},
    encode_kwargs={'normalize_embeddings': True},
)
llm_name = "mistralai/Mistral-7B-Instruct-v0.3"
TOKENIZER = AutoTokenizer.from_pretrained(llm_name)
LLM = AutoModelForCausalLM.from_pretrained(
    llm_name,
    device_map="auto",
    torch_dtype="auto"
)

if __name__ == '__main__':
    context = RagContext("dop", EMBEDDER, LLM, TOKENIZER)
    #context.add_file(Path("./res/Программа визита ГПН-Шельф в ТМК НТЦ.docx"))
    print(context.new_question("во сколько будет выступать Тюльдин?"))
