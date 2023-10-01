import logging
import os
import re
import sys

import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.callbacks import WandbCallbackHandler, CallbackManager, LlamaDebugHandler
from llama_index.query_engine import RouterQueryEngine
from llama_index.selectors.pydantic_selectors import PydanticMultiSelector
from llama_index.tools import QueryEngineTool
from llama_index.vector_stores import MilvusVectorStore
from requests.exceptions import ChunkedEncodingError
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

PROJECT_NAME = 'datatalks-faq-slackbot'
ML_ZOOMCAMP_PROJECT_NAME = 'ml-zoomcamp-slack-bot'

MLOPS_INDEX_NAME = 'mlops-faq-bot'
ML_FAQ_COLLECTION_NAME = 'mlzoomcamp_faq_git'
ML_SLACK_COLLECTION_NAME = 'mlzoomcamp_slack'

ML_FAQ_TOOL_DESCRIPTION = "Useful for retrieving specific context from the course FAQ document"
ML_SLACK_TOOL_DESCRIPTION = ("Useful for retrieving specific context from the course "
                             "slack channel history when nothing is found in the FAQ document")

# Event API & Web API
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
app = App(token=SLACK_BOT_TOKEN)


# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body):
    channel_id = body["event"]["channel"]
    event_ts = body["event"]["event_ts"]

    if channel_id not in [ML_CHANNEL_ID, MLOPS_CHANNEL_ID, TEST_WS_CHANNEL_ID, TEST_CHANNEL_ID]:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text="Apologies, I can't answer questions in this channel")
        return

    # Extract question from the message text
    question = remove_mentions(str(body["event"]["text"]))
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
            response = ml_query_engine.query(question)

        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=f"Here you go: \n{response}")
    except Exception as e:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=f"There was an error: {e}")


def remove_mentions(input_text):
    # Define a regular expression pattern to match the mention
    mention_pattern = r'<@U[0-9A-Z]+>'

    return re.sub(mention_pattern, '', input_text)


def get_greeting_message(channel_id):
    message_template = "Hello from {name}FAQ Bot! :robot_face: \n" \
                       "Please note that this is an alpha version " \
                       "and the answers might not be accurate since I'm " \
                       "just a human-friendly interface to the {name} Zoomcamp FAQ " \
                       "document that can be found in the " \
                       "<https://docs.google.com/document/d/{link}|following link>" \
                       " {additional_message}." \
                       "\nThanks for your request, I'm on it!"
    if channel_id == MLOPS_CHANNEL_ID:
        name = 'MLOps'
        link = '12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit#heading=h.uwpp1jrsj0d'
        additional_message = "and also this course's github repository"
    else:
        name = 'ML'
        link = '1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit#heading=h.98qq6wfuzeck'
        additional_message = 'and also this Slack channel'
    return message_template.format(name=name, link=link, additional_message=additional_message)


def log_langchain_to_wandb():
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


def get_query_engine_tool_by_name(collection_name, service_context, description, similarity_top_k=4):
    vector_store = MilvusVectorStore(collection_name=collection_name,
                                     uri=os.getenv("ZILLIZ_CLOUD_URI"),
                                     token=os.getenv("ZILLIZ_CLOUD_API_KEY"),
                                     dim=embedding_dimension,
                                     overwrite=False)
    vector_store_index = VectorStoreIndex.from_vector_store(vector_store,
                                                            service_context=service_context)

    return QueryEngineTool.from_defaults(
        query_engine=vector_store_index.as_query_engine(similarity_top_k=similarity_top_k),
        description=description,
    )


def init_llama_index_callback_manager():
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    wandb_callback = WandbCallbackHandler(run_args=dict(project=ML_ZOOMCAMP_PROJECT_NAME))
    return CallbackManager([wandb_callback, llama_debug])


def get_ml_query_engine():
    callback_manager = init_llama_index_callback_manager()
    service_context = ServiceContext.from_defaults(embed_model=embeddings,
                                                   callback_manager=callback_manager,
                                                   llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0.4))
    faq_tool = get_query_engine_tool_by_name(collection_name=ML_FAQ_COLLECTION_NAME,
                                             service_context=service_context,
                                             description=ML_FAQ_TOOL_DESCRIPTION)

    slack_tool = get_query_engine_tool_by_name(collection_name=ML_SLACK_COLLECTION_NAME,
                                               service_context=service_context,
                                               description=ML_SLACK_TOOL_DESCRIPTION,
                                               similarity_top_k=10)

    # Create the multi selector query engine
    return RouterQueryEngine(
        selector=PydanticMultiSelector.from_defaults(verbose=True),
        query_engine_tools=[
            slack_tool,
            faq_tool,
        ],
        service_context=service_context
    )


if __name__ == "__main__":
    client = WebClient(SLACK_BOT_TOKEN)

    logger.info('Downloading embeddings...')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    while True:
        try:
            embeddings = HuggingFaceEmbeddings()
            embedding_dimension = len(embeddings.embed_query("test"))
        except ChunkedEncodingError as e:
            continue
        break

    log_langchain_to_wandb()
    log_to_langsmith()

    mlops_index = setup_mlops_index()

    mlops_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        retriever=mlops_index.as_retriever()
    )

    ml_query_engine = get_ml_query_engine()

    SocketModeHandler(app, SLACK_APP_TOKEN).start()
