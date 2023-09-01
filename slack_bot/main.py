import os
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt import App
from slack_sdk.web import WebClient
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
import pinecone

# Event API & Web API
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "mlops-faq-slackbot"
print('Downloading embeddings...')
embeddings = HuggingFaceEmbeddings()
print('Initiating pinecone client...')
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)
index_name = 'mlops-faq-bot'
pinecone_index = Pinecone.from_existing_index(index_name=index_name,
                                              embedding=embeddings)
index = pinecone.GRPCIndex(index_name)
print(f"index stats: {index.describe_index_stats()}")

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
    retriever=pinecone_index.as_retriever()
)


# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body, logger):
    # Log message
    print(str(body["event"]["text"]).split(">")[1])

    # Create prompt for ChatGPT
    question = str(body["event"]["text"]).split(">")[1]

    # Let the user know that we are busy with the request
    client.chat_postMessage(channel=body["event"]["channel"],
                            thread_ts=body["event"]["event_ts"],
                            text=f"Hello from MLOpsFAQ Bot! :robot_face: \n"
                                 "Please note that this is an alpha version "
                                 "and the answers might not be accurate since I'm "
                                 "just a human-friendly interface to the MLOps Zoomcamp FAQ "
                                 "document that can be found in the "
                                 "<https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit#heading=h.uwpp1jrsj0d|following link>"
                                 "\nThanks for your request, I'm on it!")
    try:
        client.chat_postMessage(channel=body["event"]["channel"],
                                thread_ts=body["event"]["event_ts"],
                                text=f"Here you go: \n{qa.run(question)}")
    except Exception as e:
        client.chat_postMessage(channel=body["event"]["channel"],
                                thread_ts=body["event"]["event_ts"],
                                text=f"There was an error: {e}")


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
