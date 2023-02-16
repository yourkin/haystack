# The benchmarks use
# - a variant of the Natural Questions Dataset (https://ai.google.com/research/NaturalQuestions) from Google Research
#   licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
# - the SQuAD 2.0 Dataset (https://rajpurkar.github.io/SQuAD-explorer/) from  Rajpurkar et al.
#   licensed under  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/legalcode)
import os
import argparse
import json
import logging

import pandas as pd

from retriever import benchmark_retriever
from reader import benchmark_reader

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--reader-configs", default=None, action="store", help="Directory containing config files for reader benchmarks"
)
parser.add_argument(
    "--retriever-configs",
    default=None,
    action="store",
    help="Directory containing config files for retriever benchmarks",
)
parser.add_argument(
    "--result-file", default="benchmarking_results.json", action="store", help="File to store benchmarking results"
)

parser.add_argument(
    "--save-markdown", default=False, action="store_true", help="Whether to save the results in markdown files"
)

args = parser.parse_args()

results = {}

if args.reader_configs is not None:
    reader_results = {}
    for file in os.scandir(args.reader_configs):
        if file.is_file() and file.name.endswith(".yml"):
            logging.info(f"Running benchmark for {file.name}")
            config_id = file.name.split(".")[0]
            reader_results[config_id] = benchmark_reader(file.path)
    results["reader"] = reader_results

    if args.save_markdown:
        reader_records = [result for result in reader_results.values()]
        reader_df = pd.DataFrame.from_records(reader_records)

        with open("reader_results.md", "w") as f:
            reader_df.to_markdown(f, index=False)

if args.retriever_configs is not None:
    retriever_results = {}
    for file in os.scandir(args.retriever_configs):
        if file.is_file() and file.name.endswith(".yml"):
            logging.info(f"Running benchmark for {file.name}")
            config_id = file.name.split(".")[0]
            retriever_results[config_id] = benchmark_retriever(file.path)
    results["retriever"] = retriever_results

    if args.save_markdown:
        indexing_records = [result["indexing"] for result in results["retriever"].values()]
        indexing_df = pd.DataFrame.from_records(indexing_records)
        indexing_df = indexing_df.sort_values(by="retriever").sort_values(by="doc_store")
        with open("retriever_index_results.md", "w") as f:
            indexing_df.to_markdown(f, index=False)

        querying_records = [result["querying"] for result in results["retriever"].values()]
        querying_df = pd.DataFrame.from_records(querying_records)
        querying_df = querying_df.sort_values(by="retriever").sort_values(by="doc_store")
        with open("retriever_query_results.md", "w") as f:
            querying_df.to_markdown(f, index=False)

with open(args.result_file, "w") as f:
    json.dump(results, f, indent=4)
