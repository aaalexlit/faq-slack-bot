import datetime
import logging
import os
import re
import sys

import pinecone
from cohere import CohereAPIError
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from llama_index import ServiceContext, VectorStoreIndex, get_response_synthesizer, ChatPromptTemplate
from llama_index.callbacks import WandbCallbackHandler, CallbackManager, LlamaDebugHandler
from llama_index.indices.postprocessor import (
    TimeWeightedPostprocessor
)
from llama_index.llms import ChatMessage, MessageRole
from llama_index.postprocessor import CohereRerank
from llama_index.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.selectors.pydantic_selectors import PydanticMultiSelector
from llama_index.tools import QueryEngineTool
from llama_index.vector_stores import MilvusVectorStore, MetadataFilters
from llama_index.vector_stores.types import ExactMatchFilter
from requests.exceptions import ChunkedEncodingError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web import WebClient

logging.basicConfig(stream=sys.stdout,
                    level=os.getenv('LOG_LEVEL', logging.INFO),
                    format='%(asctime)s %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S', )
logger = logging.getLogger(__name__)

DE_CHANNELS = ['C01FABYF2RG', 'C06CBSE16JC', 'C06BZJX8PSP']
ML_CHANNELS = ['C0288NJ5XSA', 'C05C3SGMLBB', 'C05DTQECY66']
MLOPS_CHANNELS = ['C02R98X7DS9', 'C06C1N46CQ1']

ALLOWED_CHANNELS = DE_CHANNELS + ML_CHANNELS + MLOPS_CHANNELS

PROJECT_NAME = 'datatalks-faq-slackbot'
ML_ZOOMCAMP_PROJECT_NAME = 'ml-zoomcamp-slack-bot'
DE_ZOOMCAMP_PROJECT_NAME = 'de-zoomcamp-slack-bot'

MLOPS_INDEX_NAME = 'mlops-faq-bot'
ML_FAQ_COLLECTION_NAME = 'mlzoomcamp_faq_git'
DE_FAQ_COLLECTION_NAME = 'dezoomcamp_faq_git'

GPT_MODEL_NAME = 'gpt-3.5-turbo-0613'

ML_FAQ_TOOL_DESCRIPTION = ("Useful for retrieving specific context from the course FAQ document as well as "
                           "information about course syllabus and deadlines, schedule in other words. "
                           "Also, it contains midterm and capstone project evaluation criteria. "
                           "It is recommended to always check the FAQ first and then refer to the other sources.")
ML_GITHUB_TOOL_DESCRIPTION = ("Useful for retrieving specific context from the course GitHub repository that "
                              "contains information about the code, "
                              "as well as the formulation of homework assignments.")
ML_SLACK_TOOL_DESCRIPTION = ("Useful for retrieving specific context from the course "
                             "slack channel history especially the questions about homework "
                             "or when it's not likely to appear in the FAQ document. Also, Slack history can have "
                             "answers to any question, so it's always worth looking it up.")

# Event API & Web API
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
app = App(token=SLACK_BOT_TOKEN)


# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body):
    channel_id = body["event"]["channel"]
    event_ts = body["event"]["event_ts"]
    user = body["event"]["user"]

    if channel_id not in ALLOWED_CHANNELS:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text="Apologies, I can't answer questions in this channel")
        return

    # Extract question from the message text
    question = remove_mentions(str(body["event"]["text"]))
    if question.strip() == '':
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=('Ooops! It seems like your question is empty. '
                                      'Please make sure to tag me in your message along with your question.')
                                )
        return
    logger.info(question)

    # Let the user know that we are busy with the request
    greeting_message = get_greeting_message(channel_id)

    client.chat_postMessage(channel=channel_id,
                            thread_ts=event_ts,
                            text=greeting_message,
                            unfurl_links=False)
    try:
        if channel_id in MLOPS_CHANNELS:
            response = mlops_qa.run(question)
        elif channel_id in ML_CHANNELS:
            response = ml_query_engine.query(question)
        else:
            response = de_query_engine.query(question)

        response_text = f"Hey, <@{user}>! Here you go: \n{response}"

        if hasattr(response, "source_nodes"):
            sources = links_to_source_nodes(response)
            response_text += f"\nReferences:\n{sources}"

        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=response_text
                                )
    except CohereAPIError:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text="There was an error, please try again later")
    except Exception as e:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text=f"There was an error: {e}")


def links_to_source_nodes(response):
    res = set()
    source_nodes = response.source_nodes
    link_template = 'https://datatalks-club.slack.com/archives/{}/p{}'
    for node in source_nodes:
        # Slack
        if 'channel' in node.metadata:
            channel_id = node.metadata['channel']
            thread_ts = node.metadata['thread_ts']
            thread_ts_str = str(thread_ts).replace('.', '')
            link_template.format(channel_id, thread_ts_str)
            res.add(link_template.format(channel_id, thread_ts_str))
        # Google doc
        elif 'source' in node.metadata:
            title = node.metadata['title']
            if title == 'FAQ':
                section_title = node.text.split('\n', 1)[0]
                res.add(f"<{node.metadata['source']}|"
                        f" {title}-{section_title}...> ")
            else:
                res.add(f"<{node.metadata['source']}| {title}>")
        # GitHub
        elif 'repo' in node.metadata:
            repo = node.metadata['repo']
            owner = node.metadata['owner']
            branch = node.metadata['branch']
            file_path = node.metadata['file_path']
            link_to_file = build_repo_path(owner=owner, repo=repo, branch=branch, file_path=file_path)
            res.add(f'<{link_to_file}| GitHub-{repo}-{file_path.split("/")[-1]}>')
    return '\n'.join(res)


def build_repo_path(owner: str, repo: str, branch: str, file_path: str):
    return f'https://github.com/{owner}/{repo}/blob/{branch}/{file_path}'


def remove_mentions(input_text):
    # Define a regular expression pattern to match the mention
    mention_pattern = r'<@U[0-9A-Z]+>'

    return re.sub(mention_pattern, '', input_text)


def get_greeting_message(channel_id):
    message_template = "Hello from {name} FAQ Bot! :robot_face: \n" \
                       "Please note that I'm under active development. " \
                       "The answers might not be accurate since I'm " \
                       "just a human-friendly interface to the " \
                       "<https://docs.google.com/document/d/{link}| {name} Zoomcamp FAQ>" \
                       "{additional_message} and this course's <https://github.com/DataTalksClub/{repo}|GitHub repo>." \
                       "\nThanks for your request, I'm on it!"
    additional_message = ", this Slack channel,"
    if channel_id in MLOPS_CHANNELS:
        name = 'MLOps'
        link = '12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit#heading=h.uwpp1jrsj0d'
        repo = 'mlops-zoomcamp'
        additional_message = ""
    elif channel_id in ML_CHANNELS:
        name = 'ML'
        link = '1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit#heading=h.98qq6wfuzeck'
        repo = 'machine-learning-zoomcamp'
    else:
        name = 'DE'
        link = '19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit#heading=h.o29af0z8xx88'
        repo = 'data-engineering-zoomcamp'
    return message_template.format(name=name, link=link, repo=repo, additional_message=additional_message)


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


def get_query_engine_tool_by_name(collection_name: str,
                                  service_context: ServiceContext,
                                  description: str,
                                  route: str = None,
                                  similarity_top_k: int = 4,
                                  rerank_top_n: int = 2,
                                  rerank_by_time: bool = False):
    if os.getenv('LOCAL_MILVUS', None):
        localhost = os.getenv('LOCALHOST', 'localhost')
        vector_store = MilvusVectorStore(collection_name=collection_name,
                                         dim=embedding_dimension,
                                         overwrite=False,
                                         uri=f'http://{localhost}:19530')
    else:
        vector_store = MilvusVectorStore(collection_name=collection_name,
                                         uri=os.getenv("ZILLIZ_CLOUD_URI"),
                                         token=os.getenv("ZILLIZ_CLOUD_API_KEY"),
                                         dim=embedding_dimension,
                                         overwrite=False)
    vector_store_index = VectorStoreIndex.from_vector_store(vector_store,
                                                            service_context=service_context)

    cohere_rerank = CohereRerank(api_key=os.getenv('COHERE_API_KEY'), top_n=rerank_top_n)
    node_postprocessors = [cohere_rerank]
    if rerank_by_time:
        key = 'thread_ts'
        recency_postprocessor = TimeWeightedPostprocessor(
            last_accessed_key=key,
            time_decay=0.4,
            time_access_refresh=False,
            top_k=10
        )
        node_postprocessors.insert(0, recency_postprocessor)

    if route:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="route", value=route)]
        )
    else:
        filters = None

    return QueryEngineTool.from_defaults(
        query_engine=vector_store_index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=node_postprocessors,
            filters=filters,
        ),
        description=description,
        name=route
    )


def init_llama_index_callback_manager(project_name):
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    wandb_callback = WandbCallbackHandler(run_args=dict(project=project_name))
    return CallbackManager([wandb_callback, llama_debug])


def get_ml_router_query_engine():
    callback_manager = init_llama_index_callback_manager(ML_ZOOMCAMP_PROJECT_NAME)
    # Set llm temperature to 0.7 for generation
    service_context = ServiceContext.from_defaults(embed_model=embeddings,
                                                   callback_manager=callback_manager,
                                                   llm=ChatOpenAI(model=GPT_MODEL_NAME,
                                                                  temperature=0.7))
    faq_tool = get_query_engine_tool_by_name(collection_name=ML_FAQ_COLLECTION_NAME,
                                             service_context=service_context,
                                             description=ML_FAQ_TOOL_DESCRIPTION,
                                             route='faq',
                                             )

    github_tool = get_query_engine_tool_by_name(collection_name=ML_FAQ_COLLECTION_NAME,
                                                service_context=service_context,
                                                description=ML_GITHUB_TOOL_DESCRIPTION,
                                                similarity_top_k=6,
                                                rerank_top_n=3,
                                                route='github',
                                                )

    slack_tool = get_query_engine_tool_by_name(collection_name=ML_FAQ_COLLECTION_NAME,
                                               service_context=service_context,
                                               description=ML_SLACK_TOOL_DESCRIPTION,
                                               similarity_top_k=20,
                                               rerank_top_n=3,
                                               rerank_by_time=True,
                                               route='slack',
                                               )

    # Create the multi selector query engine
    # Set llm temperature to 0.4 for routing
    router_service_context = ServiceContext.from_defaults(embed_model=embeddings,
                                                          callback_manager=callback_manager,
                                                          llm=ChatOpenAI(model=GPT_MODEL_NAME,
                                                                         temperature=0.4))
    return RouterQueryEngine(
        selector=PydanticMultiSelector.from_defaults(verbose=True),
        query_engine_tools=[
            slack_tool,
            faq_tool,
            github_tool
        ],
        service_context=router_service_context
    )


def get_prompt_template(zoomcamp_name: str, cohort_year: int, course_start_date: str) -> ChatPromptTemplate:
    system_prompt = ChatMessage(
        content=(
            "You are a helpful AI assistant for the {zoomcamp_name} ZoomCamp course at DataTalksClub, "
            "and you can be found in the course's Slack channel.\n"
            "As a trustworthy assistant, you must provide helpful answers to students' questions about the course, "
            "and assist them in finding solutions when they encounter problems/errors while following the course. \n"
            "You must do it using only the excerpts from the course FAQ document, Slack threads, and GitHub repository "
            "that are provided to you, without relying on prior knowledge.\n"
            "Current cohort is year {cohort_year} one and the course start date is {course_start_date}. \n"
            "Today is {current_date}. Take this into account when answering questions with temporal aspect. \n"
            "Here are your guidelines:\n"
            "- Provide clear and concise explanations for your conclusions, including relevant evidences, and "
            "relevant code snippets if the question pertains to code. "
            "- Don't start your answer with 'Based on the provided ...' or 'The context information ...' "
            "or anything like this.\n"
            "- Justify your response in detail by explaining why you made the conclusions you actually made.\n"
            "- In your response, refrain from rephrasing the user's question or problem; simply provide an answer.\n"
            "- Make sure that the code examples you provide are accurate and runnable.\n"
            "- If the question requests confirmation, avoid repeating the question. Instead, conduct your own "
            "analysis based on the provided sources.\n"
            "- In cases where the provided information is insufficient and you are uncertain about the response, "
            "reply with: 'I don't think I have an answer for this; you'll have to ask your fellows or instructors.\n"
            "- All the hyperlinks need to be taken from the provided excerpts, not from the prior knowledge. "
            "If there are no hyperlinks provided, don't add hyperlinks to the answer.\n"
            "- The hyperlinks need to be formatted the following way: <hyperlink|displayed text> \n"
            "Example of the correctly formatted link to github: \n"
            "<https://github.com/DataTalksClub/data-engineering-zoomcamp|DE zoomcamp GitHub repo>"
        ),
        role=MessageRole.SYSTEM,
    )
    user_prompt = ChatMessage(content=("Excerpts from the course FAQ document, Slack threads, and "
                                       "GitHub repository are below delimited by the dashed lines:\n"
                                       "---------------------\n"
                                       "{context_str}\n"
                                       "---------------------\n"
                                       # "Given the information above and not prior knowledge, "
                                       # "answer the question.\n"
                                       "Question: {query_str}\n"
                                       "Answer: "),
                              role=MessageRole.USER, )
    return ChatPromptTemplate(message_templates=[
        system_prompt,
        user_prompt,
    ],
        function_mappings={'zoomcamp_name': lambda **kwargs: zoomcamp_name,
                           'cohort_year': lambda **kwargs: cohort_year,
                           'current_date': lambda **kwargs: datetime.datetime.now().strftime("%d %B %Y"),
                           'course_start_date': lambda **kwargs: course_start_date})


def get_retriever_query_engine(collection_name: str,
                               zoomcamp_name: str,
                               cohort_year: int,
                               course_start_date: str):
    callback_manager = init_llama_index_callback_manager(ML_ZOOMCAMP_PROJECT_NAME)

    service_context = ServiceContext.from_defaults(embed_model=embeddings,
                                                   callback_manager=callback_manager,
                                                   llm=ChatOpenAI(model=GPT_MODEL_NAME,
                                                                  temperature=0.7))
    if os.getenv('LOCAL_MILVUS', None):
        localhost = os.getenv('LOCALHOST', 'localhost')
        vector_store = MilvusVectorStore(collection_name=collection_name,
                                         dim=embedding_dimension,
                                         overwrite=False,
                                         uri=f'http://{localhost}:19530')
    else:
        vector_store = MilvusVectorStore(collection_name=collection_name,
                                         uri=os.getenv("ZILLIZ_CLOUD_URI"),
                                         token=os.getenv("ZILLIZ_CLOUD_API_KEY"),
                                         dim=embedding_dimension,
                                         overwrite=False)
    vector_store_index = VectorStoreIndex.from_vector_store(vector_store,
                                                            service_context=service_context)
    cohere_rerank = CohereRerank(api_key=os.getenv('COHERE_API_KEY'), top_n=4)
    recency_postprocessor = get_time_weighted_postprocessor()
    node_postprocessors = [recency_postprocessor, cohere_rerank]
    qa_prompt_template = get_prompt_template(zoomcamp_name=zoomcamp_name,
                                             cohort_year=cohort_year,
                                             course_start_date=course_start_date)
    response_synthesizer = get_response_synthesizer(service_context=service_context,
                                                    text_qa_template=qa_prompt_template,
                                                    verbose=True,
                                                    )
    return RetrieverQueryEngine(vector_store_index.as_retriever(similarity_top_k=10),
                                node_postprocessors=node_postprocessors,
                                response_synthesizer=response_synthesizer,
                                callback_manager=callback_manager)


def get_time_weighted_postprocessor():
    return TimeWeightedPostprocessor(
        last_accessed_key='thread_ts',
        time_decay=0.4,
        time_access_refresh=False,
        top_k=10,
    )


if __name__ == "__main__":
    client = WebClient(SLACK_BOT_TOKEN)

    logger.info('Downloading embeddings...')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    while True:
        try:
            embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
            embedding_dimension = len(embeddings.embed_query("test"))
        except ChunkedEncodingError as e:
            continue
        break

    log_to_langsmith()

    mlops_index = setup_mlops_index()

    mlops_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=GPT_MODEL_NAME),
        retriever=mlops_index.as_retriever()
    )

    ml_query_engine = get_retriever_query_engine(collection_name=ML_FAQ_COLLECTION_NAME,
                                                 zoomcamp_name='Machine Learning',
                                                 cohort_year=2023,
                                                 course_start_date='11 September 2023')

    de_query_engine = get_retriever_query_engine(collection_name=DE_FAQ_COLLECTION_NAME,
                                                 zoomcamp_name='Data Engineering',
                                                 cohort_year=2024,
                                                 course_start_date='15 January 2024')
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
