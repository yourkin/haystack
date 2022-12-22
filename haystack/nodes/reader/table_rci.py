from typing import List, Optional, Tuple, Dict, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from haystack.schema import Document, Answer
from haystack.nodes.reader.base import BaseReader
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.reader.table import _flatten_inputs, _calculate_answer_offsets, _check_documents

torch_scatter_installed = True
torch_scatter_wrong_version = False
try:
    import torch_scatter  # pylint: disable=unused-import
except ImportError:
    torch_scatter_installed = False
except OSError:
    torch_scatter_wrong_version = True


logger = logging.getLogger(__name__)


class RCIReader(BaseReader):
    """
    Table Reader model based on Glass et al. (2021)'s Row-Column-Intersection model.
    See the original paper for more details:
    Glass, Michael, et al. (2021): "Capturing Row and Column Semantics in Transformer Based Question Answering over Tables"
    (https://aclanthology.org/2021.naacl-main.96/)

    Each row and each column is given a score with regard to the query by two separate models. The score of each cell
    is then calculated as the sum of the corresponding row score and column score. Accordingly, the predicted answer is
    the cell with the highest score.

    Pros and Cons of RCIReader compared to TableReader:
    + Provides meaningful confidence scores
    + Allows larger tables as input
    - Does not support aggregation over table cells
    - Slower
    """

    def __init__(
        self,
        row_model_name_or_path: str = "michaelrglass/albert-base-rci-wikisql-row",
        column_model_name_or_path: str = "michaelrglass/albert-base-rci-wikisql-col",
        row_model_version: Optional[str] = None,
        column_model_version: Optional[str] = None,
        row_tokenizer: Optional[str] = None,
        column_tokenizer: Optional[str] = None,
        use_gpu: bool = True,
        top_k: int = 10,
        max_seq_len: int = 256,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Load an RCI model from Transformers.
        Available models include:

        - ``'michaelrglass/albert-base-rci-wikisql-row'`` + ``'michaelrglass/albert-base-rci-wikisql-col'``
        - ``'michaelrglass/albert-base-rci-wtq-row'`` + ``'michaelrglass/albert-base-rci-wtq-col'``


        :param row_model_name_or_path: Directory of a saved row scoring model or the name of a public model
        :param column_model_name_or_path: Directory of a saved column scoring model or the name of a public model
        :param row_model_version: The version of row model to use from the HuggingFace model hub.
                                  Can be tag name, branch name, or commit hash.
        :param column_model_version: The version of column model to use from the HuggingFace model hub.
                                     Can be tag name, branch name, or commit hash.
        :param row_tokenizer: Name of the tokenizer for the row model (usually the same as model)
        :param column_tokenizer: Name of the tokenizer for the column model (usually the same as model)
        :param use_gpu: Whether to use GPU or CPU. Falls back on CPU if no GPU is available.
        :param top_k: The maximum number of answers to return
        :param max_seq_len: Max sequence length of one input table for the model. If the number of tokens of
                            query + table exceed max_seq_len, the table will be truncated by removing rows until the
                            input size fits the model.
        :param use_auth_token:  The API token used to download private models from Huggingface.
                                If this parameter is set to `True`, then the token generated when running
                                `transformers-cli login` (stored in ~/.huggingface) will be used.
                                Additional information can be found here
                                https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )

        self.row_model = AutoModelForSequenceClassification.from_pretrained(
            row_model_name_or_path, revision=row_model_version, use_auth_token=use_auth_token
        )
        self.column_model = AutoModelForSequenceClassification.from_pretrained(
            row_model_name_or_path, revision=column_model_version, use_auth_token=use_auth_token
        )
        self.row_model.to(str(self.devices[0]))
        self.column_model.to(str(self.devices[0]))

        if row_tokenizer is None:
            try:
                self.row_tokenizer = AutoTokenizer.from_pretrained(
                    row_model_name_or_path, use_auth_token=use_auth_token
                )
            # The existing RCI models on the model hub don't come with tokenizer vocab files.
            except TypeError:
                self.row_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_auth_token=use_auth_token)
        else:
            self.row_tokenizer = AutoTokenizer.from_pretrained(row_tokenizer, use_auth_token=use_auth_token)

        if column_tokenizer is None:
            try:
                self.column_tokenizer = AutoTokenizer.from_pretrained(
                    column_model_name_or_path, use_auth_token=use_auth_token
                )
            # The existing RCI models on the model hub don't come with tokenizer vocab files.
            except TypeError:
                self.column_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_auth_token=use_auth_token)
        else:
            self.column_tokenizer = AutoTokenizer.from_pretrained(column_tokenizer, use_auth_token=use_auth_token)

        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.return_no_answers = False

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict:
        """
        Use loaded RCI models to find answers for a query in the supplied list of Documents
        of content_type ``'table'``.

        Returns dictionary containing query and list of Answer objects sorted by (desc.) score.
        The existing RCI models on the HF model hub don"t allow aggregation, therefore, the answer will always be
        composed of a single cell.

        :param query: Query string
        :param documents: List of Document in which to search for the answer. Documents should be
                          of content_type ``'table'``.
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k

        answers = []
        table_documents = _check_documents(documents)
        for document in table_documents:
            # Create row and column representations
            table: pd.DataFrame = document.content
            table = table.astype(str)
            row_reps, column_reps = self._create_row_column_representations(table)

            # Get row logits
            row_inputs = self.row_tokenizer(
                [(query, row_rep) for row_rep in row_reps],
                max_length=self.max_seq_len,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True,
            )
            row_inputs.to(self.devices[0])
            with torch.no_grad():
                row_logits = self.row_model(**row_inputs)[0].cpu().numpy()[:, 1]

            # Get column logits
            column_inputs = self.column_tokenizer(
                [(query, column_rep) for column_rep in column_reps],
                max_length=self.max_seq_len,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True,
            )
            column_inputs.to(self.devices[0])
            with torch.no_grad():
                column_logits = self.column_model(**column_inputs)[0].cpu().numpy()[:, 1]

            # Calculate cell scores
            current_answers: List[Answer] = []
            cell_scores_table: List[List[float]] = []
            for row_idx, row_score in enumerate(row_logits):
                cell_scores_table.append([])
                for col_idx, col_score in enumerate(column_logits):
                    current_cell_score = float(row_score + col_score)
                    cell_scores_table[-1].append(current_cell_score)

                    answer_str = table.iloc[row_idx, col_idx]
                    answer_offsets = _calculate_answer_offsets([(row_idx, col_idx)], table)
                    current_answers.append(
                        Answer(
                            answer=answer_str,
                            type="extractive",
                            score=current_cell_score,
                            context=table,
                            offsets_in_document=answer_offsets,
                            offsets_in_context=answer_offsets,
                            document_id=document.id,
                        )
                    )

            # Add cell scores to Answers' meta to be able to use as heatmap
            for answer in current_answers:
                answer.meta = {"table_scores": cell_scores_table}
            answers.extend(current_answers)

        # Sort answers by score and select top-k answers
        answers = sorted(answers, reverse=True)
        answers = answers[:top_k]

        results = {"query": query, "answers": answers}

        return results

    @staticmethod
    def _create_row_column_representations(table: pd.DataFrame) -> Tuple[List[str], List[str]]:
        row_reps = []
        column_reps = []
        columns = table.columns

        for idx, row in table.iterrows():
            current_row_rep = " * ".join([header + " : " + cell for header, cell in zip(columns, row)])
            row_reps.append(current_row_rep)

        for col_name in columns:
            current_column_rep = f"{col_name} * "
            current_column_rep += " * ".join(table[col_name])
            column_reps.append(current_column_rep)

        return row_reps, column_reps

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if top_k is None:
            top_k = self.top_k

        if len(documents) > 0 and isinstance(documents[0], Document):
            single_doc_list = True
        else:
            single_doc_list = False

        inputs = _flatten_inputs(queries, documents)

        results: Dict = {"queries": queries, "answers": []}
        for query, docs in zip(inputs["queries"], inputs["docs"]):
            preds = self.predict(query=query, documents=docs, top_k=top_k)
            results["answers"].append(preds["answers"])

        # Group answers by question in case of multiple queries and single doc list
        if single_doc_list and len(queries) > 1:
            answers_per_query = int(len(results["answers"]) / len(queries))
            answers = []
            for i in range(0, len(results["answers"]), answers_per_query):
                answer_group = results["answers"][i : i + answers_per_query]
                answers.append(answer_group)
            results["answers"] = answers

        return results
