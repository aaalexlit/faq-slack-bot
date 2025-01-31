"""Slack reader."""
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from http.client import IncompleteRead
from ssl import SSLContext
from typing import Any, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)
EXCLUDED_METADATA_FIELDS = ['channel', 'thread_ts']


class SlackReader(BasePydanticReader):
    """Slack reader.

    Reads conversations from channels. If the earliest_date is provided, an
    optional latest_date can also be provided. If no latest_date is provided,
    we assume the latest date is the current timestamp.

    Args:
        slack_token (Optional[str]): Slack token. If not provided, we
            assume the environment variable `SLACK_BOT_TOKEN` is set.
        ssl (Optional[str]): Custom SSL context. If not provided, it is assumed
            there is already an SSL context available.
        earliest_date (Optional[datetime]): Earliest date from which
            to read conversations. If not provided, we read all messages.
        latest_date (Optional[datetime]): Latest date from which to
            read conversations. If not provided, defaults to current timestamp
            in combination with earliest_date.
    """

    is_remote: bool = True
    slack_token: str
    earliest_date_timestamp: Optional[float]
    latest_date_timestamp: float
    bot_user_id: Optional[str]
    not_ignore_users: Optional[list[str]] = []

    _client: Any = PrivateAttr()

    def __init__(
            self,
            slack_token: Optional[str] = None,
            ssl: Optional[SSLContext] = None,
            earliest_date: Optional[datetime] = None,
            latest_date: Optional[datetime] = None,
            earliest_date_timestamp: Optional[float] = None,
            latest_date_timestamp: Optional[float] = None,
            bot_user_id: Optional[str] = None,
            not_ignore_users: Optional[list[str]] = None
    ) -> None:
        """Initialize with parameters."""
        from slack_sdk import WebClient

        if slack_token is None:
            slack_token = os.environ["SLACK_BOT_TOKEN"]
        if slack_token is None:
            raise ValueError(
                "Must specify `slack_token` or set environment "
                "variable `SLACK_BOT_TOKEN`."
            )
        if ssl is None:
            self._client = WebClient(token=slack_token)
        else:
            self._client = WebClient(token=slack_token, ssl=ssl)
        if latest_date is not None and earliest_date is None:
            raise ValueError(
                "Must specify `earliest_date` if `latest_date` is specified."
            )
        if not_ignore_users is None:
            not_ignore_users = []
        if earliest_date is not None:
            earliest_date_timestamp = earliest_date.timestamp()
        else:
            earliest_date_timestamp = None or earliest_date_timestamp
        if latest_date is not None:
            latest_date_timestamp = latest_date.timestamp()
        else:
            latest_date_timestamp = datetime.now().timestamp() or latest_date_timestamp
        res = self._client.api_test()
        if not res["ok"]:
            raise ValueError(f"Error initializing Slack API: {res['error']}")

        super().__init__(
            slack_token=slack_token,
            earliest_date_timestamp=earliest_date_timestamp,
            latest_date_timestamp=latest_date_timestamp,
            bot_user_id=bot_user_id,
            not_ignore_users=not_ignore_users,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "SlackReader"

    def _read_message(self, channel_id: str, message_ts: str) -> Document:
        from slack_sdk.errors import SlackApiError

        """Read a message."""

        messages_text: list[str] = []
        next_cursor = None
        while True:
            try:
                # https://slack.com/api/conversations.replies
                # List all replies to a message, including the message itself.
                conversations_replies_kwargs = {
                    "channel": channel_id,
                    "ts": message_ts,
                    "cursor": next_cursor,
                }
                if self.earliest_date_timestamp is not None:
                    conversations_replies_kwargs |= {
                        "latest": str(self.latest_date_timestamp),
                        "oldest": str(self.earliest_date_timestamp),
                    }
                result = self._client.conversations_replies(
                    **conversations_replies_kwargs  # type: ignore
                )
                messages = result["messages"]
                messages_text.extend(message["text"] for message in messages if message['user'] != self.bot_user_id
                                     and message['user'] not in self.not_ignore_users)
                messages_text.extend(message["attachments"][0]["text"] for message in messages if
                                     message['user'] in self.not_ignore_users
                                     and "attachments" in message
                                     and "text" in message["attachments"][0])

                if not result["has_more"]:
                    break

                next_cursor = result["response_metadata"]["next_cursor"]
            except SlackApiError as e:
                self.sleep_on_ratelimit(e)

        return Document(text="\n\n".join(messages_text),
                        metadata={"channel": channel_id, "thread_ts": float(message_ts)},
                        excluded_embed_metadata_keys=EXCLUDED_METADATA_FIELDS,
                        excluded_llm_metadata_keys=EXCLUDED_METADATA_FIELDS
                        )

    def _read_channel(self, channel_id: str) -> list[Document]:
        from slack_sdk.errors import SlackApiError

        """Read a channel."""

        thread_documents: list[Document] = []
        next_cursor = None
        while True:
            try:
                # Call the conversations.history method using the WebClient
                # conversations.history returns the first 100 messages by default
                # These results are paginated,
                # see: https://api.slack.com/methods/conversations.history$pagination
                conversations_history_kwargs = {
                    "channel": channel_id,
                    "cursor": next_cursor,
                    "latest": str(self.latest_date_timestamp),
                }
                if self.earliest_date_timestamp is not None:
                    conversations_history_kwargs["oldest"] = str(
                        self.earliest_date_timestamp
                    )
                result = self._client.conversations_history(
                    **conversations_history_kwargs  # type: ignore
                )
                conversation_history = result["messages"]
                # Print results
                logger.info(f"{len(conversation_history)} messages found in {channel_id}")

                for message in conversation_history:
                    if self.is_for_indexing(message):
                        read_message: Document = self._read_message(channel_id, message["ts"])
                        if read_message.text != "":
                            thread_documents.append(read_message)

                if not result["has_more"]:
                    break
                next_cursor = result["response_metadata"]["next_cursor"]

            except SlackApiError as e:
                self.sleep_on_ratelimit(e)
            except IncompleteRead:
                continue

        return thread_documents

    @staticmethod
    def sleep_on_ratelimit(e):
        if e.response["error"] == "ratelimited":
            retry_after = e.response.headers["retry-after"]
            logger.error(
                f'Rate limit error reached, sleeping for: {retry_after} seconds'
            )
            time.sleep(int(retry_after) + 1)
        else:
            logger.error(f"Error parsing conversation replies: {e}")

    def is_for_indexing(self, message):
        # ignore unanswered messages
        if 'reply_count' in message:
            # if bot user id isn't specified or bot hasn't replied the message
            if not self.bot_user_id or self.bot_user_id not in message['reply_users']:
                return True
            if message['reply_users_count'] > 1:
                return True
        # even if it's a single message but from a user in un-ignore list, index it
        elif message['user'] in self.not_ignore_users:
            return True
        return False

    def load_data(self, channel_ids: list[str]) -> list[Document]:
        """Load data from the input directory.

        Args:
            channel_ids (List[str]): List of channel ids to read.
        Returns:
            List[Document]: List of documents.
        """
        results = []
        for channel_id in channel_ids:
            results.extend(self._read_channel(channel_id))
        return results


if __name__ == "__main__":
    reader = SlackReader(earliest_date=datetime.now() - timedelta(days=2),
                         bot_user_id='U05DM3PEJA2',
                         not_ignore_users=['U01S08W6Z9T'])
    for thread in reader.load_data(channel_ids=["C02R98X7DS9"]):
            logger.info(f'Text: {thread.text}')
            logger.info(f'Metadata: {thread.metadata}')
            logger.info('----------------------------')
