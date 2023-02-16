import datetime
import logging
import traceback
from pathlib import Path
from time import perf_counter
import random
from typing import Dict

from haystack import Pipeline
from haystack.document_stores import eval_data_from_json
from haystack.pipelines.config import read_pipeline_config_from_yaml
from haystack.utils import aggregate_labels, stop_service
from utils import launch_document_store, download_from_url, get_documents_from_tsv

logger = logging.getLogger(__name__)


def benchmark_retriever(pipeline_yaml: Path) -> Dict:
    pipeline_config = read_pipeline_config_from_yaml(pipeline_yaml)
    benchmark_config = pipeline_config.pop("benchmark_config", {})
    n_docs = benchmark_config.get("n_docs", 0)

    indexing_pipeline = Pipeline.load_from_config(pipeline_config, pipeline_name="indexing")
    querying_pipeline = Pipeline.load_from_config(pipeline_config, pipeline_name="querying")

    # Launch DocumentStore Docker container
    for comp in pipeline_config["components"]:
        if comp["name"] == "DocumentStore":
            doc_store_type = comp["type"]
            launch_document_store(doc_store_type, n_docs)
            break

    indexing_results = benchmark_indexing(indexing_pipeline, benchmark_config)
    querying_results = benchmark_querying(querying_pipeline, indexing_pipeline, benchmark_config)

    # Stop and remove DocumentStore Docker container
    stop_service(indexing_pipeline.components["DocumentStore"], delete_container=True)

    results = {"indexing": indexing_results, "querying": querying_results}
    return results


def benchmark_indexing(pipeline: Pipeline, benchmark_config: Dict) -> Dict:
    try:
        # Get data
        documents_file = benchmark_config["documents_file"]
        n_docs = benchmark_config.get("n_docs", None)
        docs = get_documents_from_tsv(documents_file, n_docs)

        # Run indexing
        start_time = perf_counter()
        pipeline.run_batch(documents=docs)
        end_time = perf_counter()
        indexing_time = end_time - start_time

        n_docs = len(docs)
        # BM25 Retriever is not part of indexing pipelines
        try:
            retriever_name = pipeline.components["Retriever"].type
        except KeyError:
            retriever_name = "BM25Retriever"
        results = {
            "retriever": retriever_name,
            "doc_store": pipeline.components["DocumentStore"].type,
            "n_docs": n_docs,
            "indexing_time": indexing_time,
            "docs_per_second": n_docs / indexing_time,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": None,
        }

    except Exception:
        tb = traceback.format_exc()
        # BM25 Retriever is not part of indexing pipelines
        try:
            retriever_name = pipeline.components["Retriever"].type
        except KeyError:
            retriever_name = "BM25Retriever"
        doc_store_name = pipeline.components["Retriever"].type
        logger.error(
            f"##### The following Error was raised while running indexing run: {retriever_name}, {doc_store_name}, {n_docs} docs #####"
        )
        logging.error(tb)

        results = {
            "retriever": retriever_name,
            "doc_store": doc_store_name,
            "n_docs": benchmark_config.get("n_docs", None),
            "indexing_time": 0,
            "docs_per_second": 0,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(tb),
        }

    return results


def benchmark_querying(querying_pipeline: Pipeline, indexing_pipeline: Pipeline, benchmark_config: Dict) -> Dict:
    try:
        # Get eval data
        n_docs = benchmark_config["n_docs"]
        n_queries = benchmark_config.get("n_queries", None)
        labels_file = benchmark_config["labels_file"]
        if labels_file.startswith("http"):
            download_from_url(labels_file, "data/" + labels_file.split("/")[-1])
            labels_file = "data/" + labels_file.split("/")[-1]
        gold_docs, labels = eval_data_from_json(labels_file)
        gold_docs = gold_docs[:n_docs]
        # Filter labels down to n_queries
        selected_queries = list(set(f"{x.document.id} | {x.query}" for x in labels))
        n_queries = len(selected_queries) if n_queries is None else n_queries
        random.seed(42)
        selected_queries = random.sample(selected_queries, n_queries)
        agg_labels = aggregate_labels([x for x in labels if f"{x.document.id} | {x.query}" in selected_queries])

        # Index data
        indexing_pipeline.run(documents=gold_docs)

        # Run querying
        start_time = perf_counter()
        eval_result = querying_pipeline.eval_batch(labels=agg_labels)
        end_time = perf_counter()

        metrics = eval_result.calculate_metrics()["Retriever"]
        querying_time = end_time - start_time

        results = {
            "retriever": querying_pipeline.components["Retriever"].type,
            "doc_store": querying_pipeline.components["DocumentStore"].type,
            "n_docs": n_docs,
            "n_queries": len(agg_labels),
            "retrieve_time": querying_time,
            "queries_per_second": len(agg_labels) / querying_time,
            "seconds_per_query": querying_time / len(agg_labels),
            "recall": metrics["recall_single_hit"],
            "map": metrics["map"] * 100,
            "top_k": querying_pipeline.components["Retriever"].top_k,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": None,
        }

    except Exception:
        tb = traceback.format_exc()
        retriever_name = querying_pipeline.components["Retriever"].type
        doc_store_name = querying_pipeline.components["DocumentStore"].type
        logging.error(
            f"##### The following Error was raised while running querying run: {retriever_name}, {doc_store_name}, {n_docs} docs #####"
        )
        logging.error(tb)

        results = {
            "retriever": retriever_name,
            "doc_store": doc_store_name,
            "n_docs": benchmark_config.get("n_docs", None),
            "n_queries": 0,
            "retrieve_time": 0.0,
            "queries_per_second": 0.0,
            "seconds_per_query": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "top_k": 0,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(tb),
        }

    return results
