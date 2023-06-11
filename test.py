from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from keys.sensitive_info import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings = HuggingFaceEmbeddings()


def main(question):
    faiss_store = FAISS.load_local("faiss_index", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        retriever=faiss_store.as_retriever()
    )
    print(f"Question: {question}")
    print(f"Answer: {qa.run(question)}")


if __name__ == "__main__":
    main("How can I solve connection in use problem?")
