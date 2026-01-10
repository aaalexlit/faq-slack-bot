from typing import List
from llama_index.core.schema import Document
from llama_index.readers.github import GithubRepositoryReader, GithubClient
import os
import re


EXCLUDED_LLM_METADATA_KEYS = ['source', 'title', 'section_name']
EXCLUDED_EMBED_METADATA_KEYS = ['source', 'title']


class GithubRepositoryDataReader:
    """Reader for GitHub repositories with filtering capabilities."""

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        allowed_extensions: set = None,
        filename_filter: callable = None,
        branch: str = "main"
    ):
        """
        Initialize the GitHub repository data reader.

        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            allowed_extensions: Set of allowed file extensions (e.g., {"md"})
            filename_filter: Function to filter filenames (e.g., lambda fp: "_questions" in fp.lower())
            branch: Branch to read from (default: "main")
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.allowed_extensions = allowed_extensions or {"md"}
        self.filename_filter = filename_filter or (lambda fp: True)
        self.branch = branch

    def read(self) -> List[Document]:
        """
        Read documents from the GitHub repository.

        Returns:
            List of Document objects containing the raw file contents
        """
        github_client = GithubClient(os.getenv('GH_TOKEN'), verbose=True)

        # Load documents from GitHub
        documents = GithubRepositoryReader(
            github_client=github_client,
            owner=self.repo_owner,
            repo=self.repo_name,
        ).load_data(branch=self.branch)

        # Filter documents based on extension and filename filter
        filtered_docs = []
        for doc in documents:
            file_name = doc.metadata.get('file_name', '')
            file_path = doc.metadata.get('file_path', '')

            # Check extension
            if self.allowed_extensions:
                ext = os.path.splitext(file_name)[1].lstrip('.')
                if ext not in self.allowed_extensions:
                    continue

            # Check filename filter
            if not self.filename_filter(file_path):
                continue

            filtered_docs.append(doc)

        return filtered_docs


def parse_data(documents: List[Document]) -> List[Document]:
    """
    Parse FAQ documents into individual Q&A entries.

    Expects markdown files with Q&A format where questions start with '##' or '###'.

    Args:
        documents: List of raw Document objects from GitHub

    Returns:
        List of Document objects, one per Q&A entry
    """
    faq_documents = []

    for doc in documents:
        # Extract metadata from the original document
        file_path = doc.metadata.get('file_path', '')
        file_name = doc.metadata.get('file_name', '')
        branch = doc.metadata.get('branch', 'main')
        owner = doc.metadata.get('owner', '')
        repo = doc.metadata.get('repo', '')

        # Parse the markdown content
        content = doc.text

        # Split by markdown headers (## or ###)
        # This regex matches lines starting with ## or ###
        sections = re.split(r'\n(#{2,3})\s+', content)

        # sections[0] is any preamble before the first header
        # After that, sections alternate between: header-level, header-text+body, header-level, header-text+body, ...

        current_section_name = ""

        i = 0
        while i < len(sections):
            if i == 0:
                # Preamble - skip or use for context
                i += 1
                continue

            if i + 1 >= len(sections):
                break

            header_level = sections[i]  # e.g., "##" or "###"
            content_block = sections[i + 1]  # e.g., "Question title\nAnswer content"

            # Split the content block into lines
            lines = content_block.strip().split('\n', 1)

            if len(lines) == 0:
                i += 2
                continue

            # First line is the question/section title
            question = lines[0].strip()

            # Rest is the answer
            answer = lines[1].strip() if len(lines) > 1 else ""

            # Determine if this is a section header or a question
            # We'll assume ## is a section and ### is a question
            # But the filename filter should ensure we only get _questions files
            if len(header_level) == 2:  # ## - section header
                current_section_name = question
                i += 2
                continue

            # This is a question (###)
            if question and answer:
                # Create the full text combining question and answer
                full_text = f"{question}\n\n{answer}"

                # Create source URL
                source_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{file_path}"

                # Create metadata
                metadata = {
                    'source': source_url,
                    'title': 'FAQ',
                    'section_name': current_section_name,
                    'file_path': file_path,
                    'file_name': file_name,
                    'owner': owner,
                    'repo': repo,
                    'branch': branch,
                }

                # Create a new document for this Q&A
                faq_doc = Document(
                    text=full_text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=EXCLUDED_EMBED_METADATA_KEYS,
                    excluded_llm_metadata_keys=EXCLUDED_LLM_METADATA_KEYS
                )

                faq_documents.append(faq_doc)

            i += 2

    return faq_documents
