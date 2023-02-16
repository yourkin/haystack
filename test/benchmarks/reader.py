from pathlib import Path
from time import perf_counter
from typing import Dict
import traceback
import datetime
import logging

from haystack import Pipeline
from haystack.document_stores import eval_data_from_json, InMemoryDocumentStore
from haystack.modeling.data_handler.processor import _download_extract_downstream_data
from haystack.pipelines.config import read_pipeline_config_from_yaml
from haystack.utils import aggregate_labels


def benchmark_reader(pipeline_yaml: Path) -> Dict:
    pipeline_config = read_pipeline_config_from_yaml(pipeline_yaml)
    benchmark_config = pipeline_config.pop("benchmark_config", {})

    try:
        # Get data
        labels_file = benchmark_config["labels_file"]
        if labels_file == "data/squad20/dev-v2.0.json":
            _download_extract_downstream_data(input_file=labels_file)
        docs, labels = eval_data_from_json(labels_file)
        eval_labels = aggregate_labels(labels)
        eval_docs = [[label.document for label in multi_label.labels] for multi_label in eval_labels]
        document_store = InMemoryDocumentStore()
        document_store.write_documents(docs, index="eval_document")
        document_store.write_labels(labels, index="label")

        # Load Pipeline
        pipeline = Pipeline.load_from_config(pipeline_config)

        # Run querying
        start_time = perf_counter()
        eval_results = pipeline.eval_batch(labels=eval_labels, documents=eval_docs)
        # metrics = pipeline.components["Reader"].eval_on_file(data_dir="data/squad20", test_filename="dev-v2.0.json")
        # metrics = pipeline.components["Reader"].eval(document_store=document_store)
        end_time = perf_counter()
        querying_time = end_time - start_time

        metrics = eval_results.calculate_metrics()["Reader"]

        results = {
            "EM": metrics["exact_match"],
            "f1": metrics["f1"],
            "reader_time": querying_time,
            "seconds_per_query": querying_time / len(eval_labels),
            "reader": pipeline.components["Reader"].type,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": None,
        }

    except Exception:
        tb = traceback.format_exc()
        components = {comp["name"]: comp["type"] for comp in pipeline_config["components"]}
        logging.error("##### The following Error was raised while running querying run:")
        logging.error(tb)
        results = {
            "EM": 0.0,
            "f1": 0.0,
            "reader_time": 0.0,
            "seconds_per_query": 0.0,
            "reader": components.get("Reader", "No component named 'Reader' found"),
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(tb),
        }

    return results
