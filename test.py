from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from keys.sensitive_info import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
import json

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings = HuggingFaceEmbeddings()


def main(question):
    faiss_store = FAISS.load_local("faiss_index", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        retriever=faiss_store.as_retriever()
    )
    qa.return_source_documents = True
    print(f"Question: {question}")

    result = qa.apply([question])
    for res in result:
        print(res.keys())
        print(f"Question: {res['query']}")
        print(f"Answer: {res['result']}")
        for doc in res['source_documents']:
            print("----------------------------------------------------")
            print(f"Metadata: {doc.metadata}")
            print(f"Content: {doc.page_content}")
    # print(f"Answer: {[json.dumps(resp, indent=4) for resp in result][0]}")


if __name__ == "__main__":
    main("How can I solve connection in use problem?")
