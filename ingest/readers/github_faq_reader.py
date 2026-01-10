"""
GitHub FAQ Reader for ingesting FAQ markdown files from the DataTalksClub/faq repository.

This module provides functionality to read FAQ documents from GitHub and parse them
into individual Q&A entries for indexing.
"""
import os
from typing import Callable, Optional
from llama_index.core.schema import Document
from llama_index.readers.github import GithubRepositoryReader, GithubClient


class GithubRepositoryDataReader:
    """
    Reads FAQ markdown files from a GitHub repository.

    Filters files based on allowed extensions and an optional filename filter function.
    """

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        allowed_extensions: set[str],
        filename_filter: Optional[Callable[[str], bool]] = None,
        branch: str = "main",
        filter_directories: Optional[tuple[list[str], int]] = None,
    ):
        """
        Initialize the GitHub repository data reader.

        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            allowed_extensions: Set of allowed file extensions (e.g., {"md"})
            filename_filter: Optional function to filter files by name
            branch: Git branch to read from (default: "main")
            filter_directories: Optional tuple of (directories_list, filter_type) to filter directories
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.allowed_extensions = allowed_extensions
        self.filename_filter = filename_filter
        self.branch = branch
        self.filter_directories = filter_directories

    def read(self) -> list[Document]:
        """
        Read documents from the GitHub repository.

        Returns:
            List of Document objects containing the file contents and metadata
        """
        github_client = GithubClient(os.getenv('GH_TOKEN'), verbose=True)

        # Prepare filter for file extensions
        ignore_extensions = [f".{ext}" if not ext.startswith('.') else ext
                           for ext in self.allowed_extensions]
        # We want to INCLUDE only these extensions, so we invert the logic
        # by excluding everything except these

        reader_kwargs = {
            'github_client': github_client,
            'owner': self.repo_owner,
            'repo': self.repo_name,
        }

        # Add directory filter if provided
        if self.filter_directories:
            reader_kwargs['filter_directories'] = self.filter_directories

        reader = GithubRepositoryReader(**reader_kwargs)

        documents = reader.load_data(branch=self.branch)

        # Filter by extension and filename
        filtered_docs = []
        for doc in documents:
            file_path = doc.metadata.get('file_path', '')
            file_name = doc.metadata.get('file_name', '')

            # Check extension
            has_allowed_ext = any(file_name.endswith(f".{ext}" if not ext.startswith('.') else ext)
                                 for ext in self.allowed_extensions)
            if not has_allowed_ext:
                continue

            # Check filename filter
            if self.filename_filter and not self.filename_filter(file_path):
                continue

            filtered_docs.append(doc)

        return filtered_docs


def parse_data(documents: list[Document]) -> list[Document]:
    """
    Parse FAQ markdown documents into individual Q&A entries.

    The markdown format is expected to have:
    - ## for section headers (course/module names)
    - ### for question headers
    - The text after ### is the question, and the content below is the answer

    Args:
        documents: List of Document objects containing FAQ markdown content

    Returns:
        List of Document objects, each containing a single Q&A pair
    """
    faq_documents = []

    for doc in documents:
        content = doc.text
        lines = content.split('\n')

        current_section = ""
        current_question = ""
        current_answer_lines = []

        for line in lines:
            line = line.strip()

            # Section header (##)
            if line.startswith('## '):
                # Save previous Q&A if exists
                if current_question and current_answer_lines:
                    answer = '\n'.join(current_answer_lines).strip()
                    if answer:  # Only add if there's actual content
                        faq_doc = Document(
                            text=f"Question: {current_question}\n\nAnswer: {answer}",
                            metadata={
                                **doc.metadata,
                                'section_name': current_section,
                                'question': current_question,
                            }
                        )
                        faq_documents.append(faq_doc)

                # Start new section
                current_section = line[3:].strip()
                current_question = ""
                current_answer_lines = []

            # Question header (###)
            elif line.startswith('### '):
                # Save previous Q&A if exists
                if current_question and current_answer_lines:
                    answer = '\n'.join(current_answer_lines).strip()
                    if answer:  # Only add if there's actual content
                        faq_doc = Document(
                            text=f"Question: {current_question}\n\nAnswer: {answer}",
                            metadata={
                                **doc.metadata,
                                'section_name': current_section,
                                'question': current_question,
                            }
                        )
                        faq_documents.append(faq_doc)

                # Start new question
                current_question = line[4:].strip()
                current_answer_lines = []

            # Answer content
            elif current_question:
                if line:  # Skip empty lines at the start of answer
                    current_answer_lines.append(line)
                elif current_answer_lines:  # Keep empty lines within answer
                    current_answer_lines.append(line)

        # Don't forget the last Q&A
        if current_question and current_answer_lines:
            answer = '\n'.join(current_answer_lines).strip()
            if answer:
                faq_doc = Document(
                    text=f"Question: {current_question}\n\nAnswer: {answer}",
                    metadata={
                        **doc.metadata,
                        'section_name': current_section,
                        'question': current_question,
                    }
                )
                faq_documents.append(faq_doc)

    return faq_documents
