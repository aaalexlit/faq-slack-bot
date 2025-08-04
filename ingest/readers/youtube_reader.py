"""YouTube reader."""

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class YoutubeReader(BasePydanticReader):

    def __init__(self) -> None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as e:
            raise ImportError(
                '`youtube_transcript_api` must be installed to use the YoutubeReader.\n'
                'Please run `pip install --upgrade youtube-transcript-api`.'
            ) from e

        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "YoutubeReader"

    def load_data(self, video_ids: list[str], tokenizer) -> list[Document]:
        from youtube_transcript_api import YouTubeTranscriptApi

        documents: list[Document] = []
        for video_id in video_ids:
            yt_title = YoutubeReader._read_title(video_id)
            current_start = None
            current_text = ""
            current_token_count = 0
            transcript_array = YouTubeTranscriptApi().fetch(video_id)

            for snippet in transcript_array:
                # Get the token count of the current snippet text
                token_count = len(tokenizer(snippet.text, truncation=False, add_special_tokens=False)['input_ids'])

                # If adding this snippet exceeds 512 tokens, finalize the current document
                if current_token_count + token_count > 512:
                    documents.append(Document(
                        text=current_text.strip(),
                        metadata=YoutubeReader._get_node_metadata(video_id, int(current_start), yt_title),
                        excluded_embed_metadata_keys=['yt_link'],
                        excluded_llm_metadata_keys=['yt_link']
                    ))

                    # Start a new chunk
                    current_start = snippet.start
                    current_text = snippet.text
                    current_token_count = token_count
                else:
                    # Concatenate to the current chunk
                    if not current_text:
                        current_start = snippet.start
                    current_text += " " + snippet.text
                    current_token_count += token_count

            # Append the last chunk if it exists
            if current_text:
                documents.append(Document(
                    text=current_text.strip(),
                    metadata=YoutubeReader._get_node_metadata(video_id, int(current_start), yt_title),
                    excluded_embed_metadata_keys=['yt_link'],
                    excluded_llm_metadata_keys=['yt_link']
                ))

        return documents

    @staticmethod
    def _get_node_metadata(video_id: str, pos: int, yt_title: str) -> dict:
        return {
            'yt_link': f"https://www.youtube.com/watch?v={video_id}&t={pos}s",
            'yt_title': yt_title
        }

    @staticmethod
    def _read_title(video_id: str) -> str:
        params = {
            "format": "json",
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
        url = "https://www.youtube.com/oembed"

        import requests
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data['title']
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return ''
