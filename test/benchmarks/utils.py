import os

from haystack import Document
from haystack.utils import launch_milvus, launch_es, launch_opensearch, launch_weaviate
from haystack.modeling.data_handler.processor import http_get

import logging
from typing import Union, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


def launch_document_store(document_store: str, n_docs: int = 0):
    java_opts = None if n_docs < 500000 else "-Xms4096m -Xmx4096m"
    if document_store == "ElasticsearchDocumentStore":
        launch_es(delete_existing=True, java_opts=java_opts)
    elif document_store == "OpenSearchDocumentStore":
        launch_opensearch(delete_existing=True, java_opts=java_opts)
    elif document_store == "MilvusDocumentStore":
        launch_milvus(delete_existing=True)
    elif document_store == "WeaviateDocumentStore":
        launch_weaviate(delete_existing=True)


def get_documents_from_tsv(documents_file: str, n_docs: Optional[int] = None) -> List[Document]:
    if documents_file.startswith("http"):
        download_from_url(documents_file, "downloaded/" + documents_file.split("/")[-1])
        documents_file = "downloaded/" + documents_file.split("/")[-1]
    with open(documents_file) as f:
        docs = []
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            if n_docs is not None and idx > n_docs:
                break
            doc_id, text, title = line.rstrip().split("\t")
            d = Document.from_dict({"content": text, "meta": {"passage_id": int(doc_id), "title": title}})
            docs.append(d)

    return docs


def download_from_url(url: str, filepath: Union[str, Path]):
    """
    Download from a url to a local file. Skip already existing files.

    :param url: Url
    :param filepath: local path where the url content shall be stored
    :return: local path of the downloaded file
    """

    logger.info("Downloading %s", url)
    # Create local folder
    folder, filename = os.path.split(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download file if not present locally
    if os.path.exists(filepath):
        logger.info("Skipping %s (exists locally)", url)
    else:
        logger.info("Downloading %s to %s", filepath)
        with open(filepath, "wb") as file:
            http_get(url=url, temp_file=file)
    return filepath
