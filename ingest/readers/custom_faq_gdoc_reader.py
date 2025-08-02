from typing import Any, Optional

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

DEFAULT_TOKEN_JSON_PATH = 'token.json'
DEFAULT_SERVICE_ACCOUNT_JSON_PATH = 'service_account.json'
DEFAULT_CREDENTIALS_JSON_PATH = 'credentials.json'

HEADING_STYLE_TEMPLATE = 'HEADING_{}'
DEFAULT_QUESTION_HEADING_STYLE_NUM = 2

EXCLUDED_LLM_METADATA_KEYS = ['source', 'title', 'section_name']
EXCLUDED_EMBED_METADATA_KEYS = ['source', 'title']

SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]


class FAQGoogleDocsReader(BasePydanticReader):
    token_json_path: str = DEFAULT_TOKEN_JSON_PATH
    service_account_json_path: str = DEFAULT_SERVICE_ACCOUNT_JSON_PATH
    credentials_json_path: str = DEFAULT_CREDENTIALS_JSON_PATH
    question_heading_style_num: int = DEFAULT_QUESTION_HEADING_STYLE_NUM
    is_remote: bool = True

    def __init__(self,
                 token_json_path: Optional[str] = DEFAULT_TOKEN_JSON_PATH,
                 service_account_json_path: Optional[str] = DEFAULT_SERVICE_ACCOUNT_JSON_PATH,
                 credentials_json_path: Optional[str] = DEFAULT_CREDENTIALS_JSON_PATH,
                 question_heading_style_num: Optional[int] = DEFAULT_QUESTION_HEADING_STYLE_NUM
                 ) -> None:
        """Initialize with parameters."""
        try:
            import google  # noqa
            import google_auth_oauthlib  # noqa
            import googleapiclient  # noqa
        except ImportError as e:
            raise ImportError(
                '`google_auth_oauthlib`, `googleapiclient` and `google` '
                'must be installed to use the GoogleDocsReader.\n'
                'Please run `pip install --upgrade google-api-python-client '
                'google-auth-httplib2 google-auth-oauthlib`.'
            ) from e
        super().__init__(token_json_path=token_json_path,
                         service_account_json_path=service_account_json_path,
                         credentials_json_path=credentials_json_path,
                         question_heading_style_num=question_heading_style_num)

    @classmethod
    def class_name(cls) -> str:
        return 'CustomGoogleDocsReader'

    def load_data(self, document_ids: list[str]) -> list[Document]:
        """Load data from the input directory.

        Args:
            document_ids (List[str]): a list of document ids.
        """
        if document_ids is None:
            raise ValueError('Must specify a "document_ids" in `load_kwargs`.')

        results = []
        for document_id in document_ids:
            docs = self._load_docs(document_id)
            results.extend(docs)
        return results

    def _load_docs(self, document_id: str) -> list[Document]:
        """Load a document from Google Docs.

        Args:
            document_id: the document id.

        Returns:
            The document text.
        """
        import googleapiclient.discovery as discovery

        credentials = self._get_credentials()
        docs_service = discovery.build('docs', 'v1', credentials=credentials)
        doc = docs_service.documents().get(documentId=document_id).execute()
        doc_content = doc.get('body').get('content')
        doc_source = f'https://docs.google.com/document/d/{document_id}/edit#heading='
        return self._structural_elements_to_docs(doc_content, doc_source)

    def _get_credentials(self) -> Any:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
        from google.auth.transport.requests import Request
        import google.auth

        creds = google.auth.default(scopes=SCOPES)[0]

        return creds

    @staticmethod
    def _read_paragraph_element(element: Any) -> Any:
        """Return the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
        """
        text_run = element.get('textRun')
        return text_run.get('content') if text_run else ''

    @staticmethod
    def _get_text_from_paragraph_elements(elements: [Any]) -> Any:
        return ''.join(FAQGoogleDocsReader._read_paragraph_element(elem) for elem in elements)

    def _structural_elements_to_docs(self,
                                     doc_elements: list[Any],
                                     doc_source: str) -> list[Document]:
        """Recurse through a list of Structural Elements.

        Read a document's text where the text may be in nested elements.

        Args:
            doc_elements: a list of Structural Elements.
        """
        docs = []
        text = ''
        heading_id = ''
        section_name = ''
        question_heading_style = HEADING_STYLE_TEMPLATE.format(self.question_heading_style_num)
        section_heading_style = HEADING_STYLE_TEMPLATE.format(self.question_heading_style_num - 1)
        for value in doc_elements:
            if 'paragraph' in value:
                paragraph = value['paragraph']
                elements = paragraph.get('elements')
                paragraph_text = FAQGoogleDocsReader._get_text_from_paragraph_elements(elements)
                if 'paragraphStyle' in paragraph and 'headingId' in paragraph['paragraphStyle']:
                    named_style_type = paragraph['paragraphStyle']['namedStyleType']
                    if named_style_type in [
                        question_heading_style,
                        section_heading_style,
                    ]:
                        # create previous document checking if it's not empty
                        if text != '':
                            node_metadata = {
                                'source': doc_source + heading_id,
                                'section_name': section_name,
                                'title': 'FAQ'
                            }
                            prev_doc = Document(text=text,
                                                metadata=node_metadata,
                                                excluded_embed_metadata_keys=EXCLUDED_EMBED_METADATA_KEYS,
                                                excluded_llm_metadata_keys=EXCLUDED_LLM_METADATA_KEYS)
                            docs.append(prev_doc)
                        if named_style_type == question_heading_style:
                            heading_id = paragraph['paragraphStyle']['headingId']
                            text = paragraph_text
                        else:
                            section_name = paragraph_text
                            text = ''
                else:
                    text += paragraph_text
        return docs


if __name__ == '__main__':
    reader = FAQGoogleDocsReader(service_account_json_path='../keys/service_account_key.json')
    docs = reader.load_data(['1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8'])
    print(docs)
