"""
GitHub-based FAQ reader for DataTalksClub/faq repository.
Based on the docs.py example provided in issue #4.
"""
import os
import re
from typing import List, Callable, Optional
from llama_index.core.schema import Document
from llama_index.readers.github import GithubRepositoryReader, GithubClient


class GithubRepositoryDataReader:
    """
    Reader for FAQ documents from GitHub repository.
    Filters files by extension and filename pattern.
    """

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        allowed_extensions: set[str],
        filename_filter: Callable[[str], bool],
        branch: str = "main",
        folder_filter: Optional[str] = None,
    ):
        """
        Initialize the GitHub repository data reader.

        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            allowed_extensions: Set of allowed file extensions (e.g., {"md"})
            filename_filter: Function to filter files by name pattern
            branch: Git branch to read from
            folder_filter: Optional folder path to filter (e.g., "_questions/ml-zoomcamp")
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.allowed_extensions = allowed_extensions
        self.filename_filter = filename_filter
        self.branch = branch
        self.folder_filter = folder_filter

    def read(self) -> List[Document]:
        """
        Read documents from the GitHub repository.

        Returns:
            List of Document objects containing file contents
        """
        github_client = GithubClient(os.getenv('GH_TOKEN'), verbose=True)

        # Build filter for file extensions
        filter_extensions = [f".{ext}" if not ext.startswith(".") else ext
                            for ext in self.allowed_extensions]

        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=self.repo_owner,
            repo=self.repo_name,
            filter_file_extensions=(
                filter_extensions,
                GithubRepositoryReader.FilterType.INCLUDE
            ),
        )

        # Load all documents from the branch
        documents = reader.load_data(branch=self.branch)

        # Apply filename filter
        filtered_docs = [
            doc for doc in documents
            if self.filename_filter(doc.metadata.get('file_path', ''))
        ]

        # Apply folder filter if specified
        if self.folder_filter:
            filtered_docs = [
                doc for doc in filtered_docs
                if self.folder_filter in doc.metadata.get('file_path', '')
            ]

        return filtered_docs


def parse_data(documents: List[Document]) -> List[Document]:
    """
    Parse FAQ documents into individual Q&A entries.

    Expected format:
    - ## Section Title
    - ### Question
    - Answer text

    Args:
        documents: List of Document objects containing FAQ markdown

    Returns:
        List of Document objects, one per Q&A pair
    """
    faq_documents = []

    for doc in documents:
        content = doc.text
        file_path = doc.metadata.get('file_path', '')
        source_url = f"https://github.com/{doc.metadata.get('owner', '')}/{doc.metadata.get('repo', '')}/blob/{doc.metadata.get('branch', 'main')}/{file_path}"

        # Extract the document title (first # or ## heading)
        title_match = re.search(r'^#+ (.+)$', content, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else file_path

        # Split content into sections by ## headings
        sections = re.split(r'\n## ', content)

        for section in sections[1:]:  # Skip first split (before first ##)
            lines = section.split('\n')
            section_name = lines[0].strip()

            # Find all questions (### headings) in this section
            questions = []
            current_question = None
            current_answer = []

            for line in lines[1:]:
                if line.startswith('### '):
                    # Save previous question if exists
                    if current_question:
                        questions.append({
                            'question': current_question,
                            'answer': '\n'.join(current_answer).strip()
                        })
                    # Start new question
                    current_question = line[4:].strip()
                    current_answer = []
                elif line.startswith('## '):
                    # New section started, save current question
                    if current_question:
                        questions.append({
                            'question': current_question,
                            'answer': '\n'.join(current_answer).strip()
                        })
                    current_question = None
                    current_answer = []
                    break
                else:
                    if current_question:
                        current_answer.append(line)

            # Save last question in section
            if current_question:
                questions.append({
                    'question': current_question,
                    'answer': '\n'.join(current_answer).strip()
                })

            # Create a document for each Q&A pair
            for qa in questions:
                faq_text = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"

                faq_doc = Document(
                    text=faq_text,
                    metadata={
                        'source': source_url,
                        'title': doc_title,
                        'section_name': section_name,
                        'question': qa['question'],
                        'file_path': file_path,
                        'owner': doc.metadata.get('owner', ''),
                        'repo': doc.metadata.get('repo', ''),
                        'branch': doc.metadata.get('branch', 'main'),
                    }
                )
                faq_documents.append(faq_doc)

    return faq_documents
