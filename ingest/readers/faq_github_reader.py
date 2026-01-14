"""GitHub FAQ reader."""
import os
from typing import Optional

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from llama_index.readers.github import GithubRepositoryReader, GithubClient

EXCLUDED_LLM_METADATA_KEYS = ['source', 'title', 'id', 'sort_order', 'file_path', 'module']
EXCLUDED_EMBED_METADATA_KEYS = ['source', 'title', 'id', 'sort_order']

# Map collection names to zoomcamp directory names
ZOOMCAMP_DIR_MAPPING = {
    'mlzoomcamp_faq_git': 'machine-learning-zoomcamp',
    'dezoomcamp_faq_git': 'data-engineering-zoomcamp',
    'mlopszoomcamp': 'mlops-zoomcamp',
    'llmzoomcamp': 'llm-zoomcamp',
}


class FAQGithubReader(BasePydanticReader):
    """
    Reader for GitHub-based FAQ markdown files from DataTalksClub/faq repository.

    Reads FAQ files with YAML frontmatter (id, question, sort_order) and creates
    LlamaIndex documents with appropriate metadata for the Slack bot.

    Args:
        github_token (Optional[str]): GitHub token. If not provided, we assume
            the environment variable `GH_TOKEN` is set.
        owner (str): Repository owner. Defaults to 'DataTalksClub'.
        repo (str): Repository name. Defaults to 'faq'.
        branch (str): Branch name. Defaults to 'main'.
    """

    is_remote: bool = True
    github_token: str
    owner: str = "DataTalksClub"
    repo: str = "faq"
    branch: str = "main"

    def __init__(
        self,
        github_token: Optional[str] = None,
        owner: str = "DataTalksClub",
        repo: str = "faq",
        branch: str = "main"
    ) -> None:
        """Initialize with GitHub token."""
        try:
            import frontmatter  # noqa
        except ImportError as e:
            raise ImportError(
                '`python-frontmatter` must be installed.\n'
                'Please run `pip install python-frontmatter`.'
            ) from e

        if github_token is None:
            github_token = os.getenv('GH_TOKEN')
            if not github_token:
                raise ValueError(
                    "GitHub token must be provided either as parameter or via GH_TOKEN environment variable"
                )

        super().__init__(
            github_token=github_token,
            owner=owner,
            repo=repo,
            branch=branch
        )

    @classmethod
    def class_name(cls) -> str:
        return 'FAQGithubReader'

    def load_data(self, zoomcamp_name: str) -> list[Document]:
        """
        Load FAQ documents for a specific zoomcamp.

        Args:
            zoomcamp_name: Name of the zoomcamp directory (e.g., 'machine-learning-zoomcamp')

        Returns:
            List of Document objects with parsed frontmatter metadata
        """
        import frontmatter

        print(f"Fetching FAQ files from GitHub: {self.owner}/{self.repo}")

        # 1. Fetch all markdown files from GitHub using LlamaIndex's GithubRepositoryReader
        github_client = GithubClient(self.github_token, verbose=True)
        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=self.owner,
            repo=self.repo,
            filter_file_extensions=(['.md'], GithubRepositoryReader.FilterType.INCLUDE),
        )

        try:
            all_docs = reader.load_data(branch=self.branch)
        except Exception as e:
            print(f"Error fetching repository data: {e}")
            raise

        print(f"Fetched {len(all_docs)} markdown files from repository")

        # 2. Filter for FAQ files in the specific zoomcamp directory
        pattern = f"_questions/{zoomcamp_name}/"
        faq_docs = [doc for doc in all_docs if pattern in doc.metadata.get('file_path', '')]

        print(f"Found {len(faq_docs)} FAQ files for {zoomcamp_name}")

        # 3. Parse frontmatter and create new documents
        result_docs = []
        for doc in faq_docs:
            try:
                # Parse YAML frontmatter
                post = frontmatter.loads(doc.text)
                metadata = post.metadata

                # Validate required fields
                if not all(k in metadata for k in ['id', 'question', 'sort_order']):
                    print(f"Warning: Missing required fields in {doc.metadata['file_path']}, skipping")
                    continue

                # Extract zoomcamp name and module from file path
                # Path format: _questions/{zoomcamp}/{module}/{file}.md
                file_path = doc.metadata['file_path']
                path_parts = file_path.split('/')
                zoomcamp_dir = path_parts[1] if len(path_parts) > 1 else zoomcamp_name
                module = path_parts[2] if len(path_parts) > 2 else 'general'

                # Build datatalks.club FAQ URL with ID anchor
                # Format: https://datatalks.club/faq/{zoomcamp}.html#{id}
                source_url = f"https://datatalks.club/faq/{zoomcamp_dir}.html#{metadata['id']}"

                # Create document with question + answer
                # Include question in text for better embedding/retrieval
                text = f"{metadata['question']}\n\n{post.content}"

                new_metadata = {
                    'source': source_url,
                    'question': metadata['question'],
                    'module': module,
                    'id': metadata['id'],
                    'sort_order': metadata.get('sort_order', 0),
                    'title': 'FAQ',
                    'file_path': file_path,
                }

                new_doc = Document(
                    text=text,
                    metadata=new_metadata,
                    excluded_embed_metadata_keys=EXCLUDED_EMBED_METADATA_KEYS,
                    excluded_llm_metadata_keys=EXCLUDED_LLM_METADATA_KEYS
                )
                result_docs.append(new_doc)

            except Exception as e:
                print(f"Error parsing {doc.metadata.get('file_path', 'unknown')}: {e}")
                continue

        print(f"Successfully loaded {len(result_docs)} FAQ documents for {zoomcamp_name}")
        return result_docs