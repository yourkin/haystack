# The benchmarks use
# - a variant of the Natural Questions Dataset (https://ai.google.com/research/NaturalQuestions) from Google Research
#   licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
# - the SQuAD 2.0 Dataset (https://rajpurkar.github.io/SQuAD-explorer/) from  Rajpurkar et al.
#   licensed under  CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/legalcode)
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from retriever import benchmark_retriever
from reader import benchmark_reader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_benchmarks(reader_configs: Path, retriever_configs: Path, save_markdown):
    results = {}

    if reader_configs is not None:
        reader_results = {}
        for file in reader_configs.glob("**/*.yml"):
            logger.info("Running benchmark for %s", file.name)
            config_id = file.stem
            reader_results[config_id] = benchmark_reader(file.absolute())
        results["reader"] = reader_results

        if save_markdown:
            reader_records = [result for result in reader_results.values()]
            reader_df = pd.DataFrame.from_records(reader_records)

            with open("reader_results.md", "w") as f:
                reader_df.to_markdown(f, index=False)

    if retriever_configs is not None:
        retriever_results = {}
        for file in retriever_configs.glob("**/*.yml"):
            logger.info("Running benchmark for %s", file.name)
            config_id = file.stem
            retriever_results[config_id] = benchmark_retriever(file.absolute())
        results["retriever"] = retriever_results

        if save_markdown:
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

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reader-configs",
        default=None,
        action="store",
        type=Path,
        help="Directory containing config files for reader benchmarks",
    )
    parser.add_argument(
        "--retriever-configs",
        default=None,
        action="store",
        type=Path,
        help="Directory containing config files for retriever benchmarks",
    )
    parser.add_argument(
        "--result-file",
        default="benchmarking_results.json",
        action="store",
        type=Path,
        help="File to store benchmarking results",
    )

    parser.add_argument(
        "--save-markdown", default=False, action="store_true", help="Whether to save the results in markdown files"
    )

    args = parser.parse_args()

    results = run_benchmarks(args.reader_configs, args.retriever_configs, args.save_markdown)

    with args.result_file.open("w", "utf-8") as f:
        json.dump(results, f, indent=4)
