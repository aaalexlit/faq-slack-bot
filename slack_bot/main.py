import datetime
import hashlib
import logging
import os
import re
import sys
import uuid

from cohere.core import ApiError as CohereAPIError
from langchain import callbacks
from langchain_openai import ChatOpenAI
from langsmith import Client
from llama_index.core import ChatPromptTemplate
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import TimeWeightedPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from requests.exceptions import ChunkedEncodingError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.views import View
from slack_sdk.web import WebClient

logging.basicConfig(stream=sys.stdout,
                    level=os.getenv('LOG_LEVEL', logging.INFO),
                    format='%(asctime)s %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S', )
logger = logging.getLogger(__name__)

DE_CHANNELS = ['C01FABYF2RG', 'C06CBSE16JC', 'C06BZJX8PSP']
ML_CHANNELS = ['C0288NJ5XSA', 'C05C3SGMLBB', 'C05DTQECY66']
MLOPS_CHANNELS = ['C02R98X7DS9', 'C06C1N46CQ1', 'C0735558X52']
LLM_CHANNELS = ['C079QE5NAMP', 'C078X7REVN3', 'C06TEGTGM3J']

ALLOWED_CHANNELS = DE_CHANNELS + ML_CHANNELS + MLOPS_CHANNELS + LLM_CHANNELS

PROJECT_NAME = 'datatalks-faq-slackbot'
ML_ZOOMCAMP_PROJECT_NAME = 'ml-zoomcamp-slack-bot'
DE_ZOOMCAMP_PROJECT_NAME = 'de-zoomcamp-slack-bot'

ML_COLLECTION_NAME = 'mlzoomcamp_faq_git'
DE_COLLECTION_NAME = 'dezoomcamp_faq_git'
MLOPS_COLLECTION_NAME = 'mlopszoomcamp'
LLM_COLLECTION_NAME = 'llmzoomcamp'

GPT_MODEL_NAME = 'gpt-4o-mini-2024-07-18'

# Event API & Web API
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
app = App(token=SLACK_BOT_TOKEN)
langsmith_client = Client()


@app.action('upvote')
def add_positive_feedback(ack, body):
    ack()
    add_feedback(body, 'upvote')


@app.action('downvote')
def add_negative_feedback(ack, body):
    ack()
    add_feedback(body, 'downvote')


def add_feedback(body, feedback_type: str):
    run_id = None
    feedback_id = None
    try:
        original_blocks = body['message']['blocks']
        actions_block_elements = [block for block in original_blocks if block.get('type') == 'actions'][0]['elements']
        element_to_update = \
            [element for element in actions_block_elements if element.get('action_id') == feedback_type][0]
        element_text_to_update = element_to_update['text']['text']
        updated_text, updated_number = increment_number_in_string(element_text_to_update)
        element_to_update['text']['text'] = updated_text

        run_id = body['actions'][0]['value']
        feedback_id = get_feedback_id_from_run_id_and_feedback_type(run_id, feedback_type)

        user_id = body['user']['id']
        user_name = body['user']['username']

        logger.info(f'run_id {run_id} {feedback_type}d by {user_name}({user_id})')

        if updated_number > 1:
            langsmith_client.update_feedback(
                feedback_id=feedback_id,
                score=updated_number
            )
        else:
            langsmith_client.create_feedback(
                run_id=run_id,
                key=feedback_type,
                score=updated_number,
                feedback_id=feedback_id
            )

        client.chat_update(
            channel=body['channel']['id'],
            ts=body['message']['ts'],
            blocks=original_blocks,
            text=body['message']['text']
        )
    except Exception as ex:
        error_message = f'An error occurred when trying to record user feedback with action body =\n{body}\n'
        if run_id:
            error_message += f'for run_id = {run_id}\n'
        if feedback_id:
            error_message += f'and feedback_id = {feedback_id}\n'

        logger.error(f'{error_message}'
                     f'Error: {ex}')
        show_feedback_logging_error_modal(body['trigger_id'])


def show_feedback_logging_error_modal(trigger_id):
    client.views_open(trigger_id=trigger_id,
                      view=View(type='modal',
                                title='Error recording feedback',
                                blocks=[
                                    {
                                        "type": "section",
                                        "text": {
                                            "type": "mrkdwn",
                                            "text": (
                                                "An error occurred while attempting to capture your feedback.\n"
                                                "Please try again later. Apologies for the inconvenience.")
                                        }
                                    }
                                ]))


def get_feedback_id_from_run_id_and_feedback_type(run_id, feedback_type):
    # Combine run_id UUID bytes and action bytes
    combined_bytes = uuid.UUID(run_id).bytes + feedback_type.encode('utf-8')
    # Hash the combined bytes
    hashed_bytes = hashlib.sha1(combined_bytes).digest()
    # Convert hashed bytes to UUID
    return uuid.UUID(bytes=hashed_bytes[:16])


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

    posted_greeting_message = client.chat_postMessage(channel=channel_id,
                                                      thread_ts=event_ts,
                                                      text=greeting_message,
                                                      unfurl_links=False)
    try:
        with callbacks.collect_runs() as cb:
            if channel_id in MLOPS_CHANNELS:
                response = mlops_query_engine.query(question)
            elif channel_id in ML_CHANNELS:
                response = ml_query_engine.query(question)
            elif channel_id in LLM_CHANNELS:
                response = llm_query_engine.query(question)
            else:
                response = de_query_engine.query(question)
            # get the id of the last run that's supposedly a run that delivers the final answer
            run_id = cb.traced_runs[-1].id

        response_text = f"Hey, <@{user}>! Here you go: \n{response}"

        response_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": response_text
                }
            },
            {
                "type": "divider"
            }]
        if hasattr(response, "source_nodes"):
            sources = links_to_source_nodes(response)
            references = f"References:\n{sources}"
            references_blocks = [{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": references
                }
            },
                {
                    "type": "divider"
                }]
            response_blocks.extend(references_blocks)

        response_blocks.extend([{
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":pray: Please leave your feedback to help me improve "
                }
            ]
        },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": ":thumbsup: 0"
                        },
                        "style": "primary",
                        "value": f"{run_id}",
                        "action_id": "upvote"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": ":thumbsdown: 0"
                        },
                        "style": "danger",
                        "value": f"{run_id}",
                        "action_id": "downvote"
                    }
                ]
            }
        ])

        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                blocks=response_blocks,
                                text=response_text,
                                unfurl_media=False
                                )
        client.chat_delete(channel=channel_id,
                           ts=posted_greeting_message.data['ts'])
    except CohereAPIError:
        client.chat_postMessage(channel=channel_id,
                                thread_ts=event_ts,
                                text="There was an error, please try again later")
    except Exception as e:
        logger.error(f'Error responding to a query\n{e}')
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
        elif 'yt_link' in node.metadata:
            yt_link = node.metadata['yt_link']
            yt_title = node.metadata['yt_title']
            res.add(f'<{yt_link}| Youtube-{yt_title}>')
    return '\n'.join(res)


def increment_number_in_string(source_string):
    # Regular expression to find any sequence of digits (\d+)
    pattern = r'(\d+)'

    # Define a lambda function to replace matched digits with the incremented value
    replacer = lambda match: str(int(match.group(0)) + 1)

    # Use re.sub() to replace matched digits with the incremented value
    result_string = re.sub(pattern, replacer, source_string)
    result_number = int(re.search(pattern, result_string).group(0))

    return result_string, result_number


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
                       ", this Slack channel, and this course's <https://github.com/DataTalksClub/{repo}|GitHub repo>." \
                       "\nThanks for your request, I'm on it!"
    if channel_id in MLOPS_CHANNELS:
        name = 'MLOps'
        link = '12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit#heading=h.uwpp1jrsj0d'
        repo = 'mlops-zoomcamp'
    elif channel_id in ML_CHANNELS:
        name = 'ML'
        link = '1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit#heading=h.98qq6wfuzeck'
        repo = 'machine-learning-zoomcamp'
    elif channel_id in LLM_CHANNELS:
        name = 'LLM'
        link = '1m2KexowAXTmexfC5rVTCSnaShvdUQ8Ag2IEiwBDHxN0/edit#heading=h.o29af0z8xx88'
        repo = 'llm-zoomcamp'
    else:
        name = 'DE'
        link = '19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit#heading=h.o29af0z8xx88'
        repo = 'data-engineering-zoomcamp'
    return message_template.format(name=name, link=link, repo=repo)


def log_to_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME


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
            "relevant code snippets if the question pertains to code. \n"
            "- Avoid starting your answer with 'Based on the provided ...' or 'The context information ...' "
            "or anything like this, instead, provide the information directly in the response.\n"
            "- Justify your response in detail by explaining why you made the conclusions you actually made.\n"
            "- In your response, refrain from rephrasing the user's question or problem; simply provide an answer.\n"
            "- Make sure that the code examples you provide are accurate and runnable.\n"
            "- If the question requests confirmation, avoid repeating the question. Instead, conduct your own "
            "analysis based on the provided sources.\n"
            "- In cases where the provided information is insufficient and you are uncertain about the response, "
            "reply with: 'I don't think I have an answer for this; you'll have to ask your fellows or instructors.\n"
            "- All the hyperlinks need to be taken from the provided excerpts, not from prior knowledge. "
            "If there are no hyperlinks provided, abstain from adding hyperlinks to the answer.\n"
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
    if os.getenv('LOCAL_MILVUS', None):
        localhost = os.getenv('LOCALHOST', 'localhost')
        vector_store = MilvusVectorStore(collection_name=collection_name,
                                         dim=embedding_dimension,
                                         overwrite=False,
                                         uri=f'http://{localhost}:19530')
    else:
        if collection_name in [MLOPS_COLLECTION_NAME, LLM_COLLECTION_NAME]:
            vector_store = MilvusVectorStore(collection_name=collection_name,
                                             uri=os.getenv("ZILLIZ_PUBLIC_ENDPOINT"),
                                             token=os.getenv("ZILLIZ_API_KEY"),
                                             dim=embedding_dimension,
                                             overwrite=False)
        else:
            vector_store = MilvusVectorStore(collection_name=collection_name,
                                             uri=os.getenv("ZILLIZ_CLOUD_URI"),
                                             token=os.getenv("ZILLIZ_CLOUD_API_KEY"),
                                             dim=embedding_dimension,
                                             overwrite=False)
    vector_store_index = VectorStoreIndex.from_vector_store(vector_store,
                                                            embed_model=embeddings)
    # cohere_rerank = CohereRerank(api_key=os.getenv('COHERE_API_KEY'), top_n=4)
    recency_postprocessor = get_time_weighted_postprocessor()
    # node_postprocessors = [recency_postprocessor, cohere_rerank]
    node_postprocessors = [recency_postprocessor]
    qa_prompt_template = get_prompt_template(zoomcamp_name=zoomcamp_name,
                                             cohort_year=cohort_year,
                                             course_start_date=course_start_date)
    Settings.llm = ChatOpenAI(model=GPT_MODEL_NAME,
                              temperature=0.7)

    response_synthesizer = get_response_synthesizer(text_qa_template=qa_prompt_template,
                                                    verbose=True,
                                                    )
    return RetrieverQueryEngine(vector_store_index.as_retriever(similarity_top_k=15),
                                node_postprocessors=node_postprocessors,
                                response_synthesizer=response_synthesizer,
                                )


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
            embeddings = HuggingFaceEmbedding(model_name='BAAI/bge-base-en-v1.5')
            embedding_dimension = len(embeddings.get_text_embedding("test"))
        except ChunkedEncodingError as e:
            continue
        break

    log_to_langsmith()

    ml_query_engine = get_retriever_query_engine(collection_name=ML_COLLECTION_NAME,
                                                 zoomcamp_name='Machine Learning',
                                                 cohort_year=2024,
                                                 course_start_date='16 September 2024')

    de_query_engine = get_retriever_query_engine(collection_name=DE_COLLECTION_NAME,
                                                 zoomcamp_name='Data Engineering',
                                                 cohort_year=2025,
                                                 course_start_date='13 January 2025')

    mlops_query_engine = get_retriever_query_engine(collection_name=MLOPS_COLLECTION_NAME,
                                                    zoomcamp_name='MLOps',
                                                    cohort_year=2024,
                                                    course_start_date='13 May 2024')

    llm_query_engine = get_retriever_query_engine(collection_name=LLM_COLLECTION_NAME,
                                                  zoomcamp_name='LLM',
                                                  cohort_year=2024,
                                                  course_start_date='17 June 2024')
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
