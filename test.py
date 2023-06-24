import os
import pinecone
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings = HuggingFaceEmbeddings()


def main(question):
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    pinecone_index = Pinecone.from_existing_index(index_name='mlops-faq-bot',
                                                  embedding=embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        retriever=pinecone_index.as_retriever()
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


if __name__ == "__main__":
    main("How can I solve connection in use problem with mlflow?")
