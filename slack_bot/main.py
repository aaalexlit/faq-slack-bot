import logging
import os
import sys

import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone, Milvus
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web import WebClient

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S', )
logger = logging.getLogger(__name__)

MLOPS_CHANNEL_ID = "C02R98X7DS9"
ML_CHANNEL_ID = "C0288NJ5XSA"
TEST_WS_CHANNEL_ID = os.getenv('TEST_WS_CHANNEL_ID', '')
TEST_CHANNEL_ID = os.getenv('TEST_CHANNEL_ID', '')

PROJECT_NAME = "datatalks-faq-slackbot"

MLOPS_INDEX_NAME = 'mlops-faq-bot'
ML_INDEX_NAME = 'mlzoomcamp'

# Event API & Web API
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
app = App(token=SLACK_BOT_TOKEN)


# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body):
    channel_id = body["event"]["channel"]
    event_ts = body["event"]["event_ts"]

    if channel_id not in [ML_CHANNEL_ID, MLOPS_CHANNEL_ID, TEST_WS_CHANNEL_ID]:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text="Apologies, I can't answer questions in this channel")
        return

    # Extract question from the message text
    question = str(body["event"]["text"]).split(">")[1]
    logger.info(question)

    # Let the user know that we are busy with the request
    greeting_message = get_greeting_message(channel_id)

    client.chat_postMessage(channel=channel_id,
                            thread_ts=event_ts,
                            text=greeting_message)
    try:
        if channel_id in [MLOPS_CHANNEL_ID]:
            response = mlops_qa.run(question)
        else:
            response = ml_qa.run(question)

        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=f"Here you go: \n{response}")
    except Exception as e:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=f"There was an error: {e}")


def get_greeting_message(channel_id):
    name = 'MLOps' if channel_id == MLOPS_CHANNEL_ID else 'ML'
    link = '12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit#heading=h.uwpp1jrsj0d' if channel_id == MLOPS_CHANNEL_ID \
        else '1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit#heading=h.98qq6wfuzeck'
    return f"Hello from {name}FAQ Bot! :robot_face: \n" \
           "Please note that this is an alpha version " \
           "and the answers might not be accurate since I'm " \
           f"just a human-friendly interface to the {name} Zoomcamp FAQ " \
           "document that can be found in the " \
           f"<https://docs.google.com/document/d/{link}|following link>" \
           "\nThanks for your request, I'm on it!"


def log_to_wandb():
    # Log everything to WANDB!!!
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = PROJECT_NAME


def log_to_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME


def setup_mlops_index():
    logger.info('Initiating pinecone client...')
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    pinecone_index = Pinecone.from_existing_index(index_name=MLOPS_INDEX_NAME,
                                                  embedding=embeddings)
    index = pinecone.GRPCIndex(MLOPS_INDEX_NAME)
    logger.info(f"Mlops index stats: {index.describe_index_stats()}")
    return pinecone_index


def setup_ml_index():
    return Milvus(embedding_function=embeddings,
                  collection_name=ML_INDEX_NAME,
                  connection_args={
                      "uri": os.getenv("ZILLIZ_CLOUD_URI"),
                      "token": os.getenv("ZILLIZ_CLOUD_API_KEY"),
                      "secure": True,
                  })


if __name__ == "__main__":
    client = WebClient(SLACK_BOT_TOKEN)

    logger.info('Downloading embeddings...')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embeddings = HuggingFaceEmbeddings()

    log_to_wandb()
    log_to_langsmith()

    mlops_index = setup_mlops_index()
    ml_index = setup_ml_index()

    mlops_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        retriever=mlops_index.as_retriever()
    )

    ml_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        retriever=ml_index.as_retriever(),
    )
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
