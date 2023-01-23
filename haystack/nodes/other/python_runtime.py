from haystack import MultiLabel, Document
from haystack.nodes.base import BaseComponent

from typing import Dict, Optional, Union, List, Any, Tuple


class PythonRuntime(BaseComponent):

    outgoing_edges = 1

    def __init__(self):
        super().__init__()

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        if not invocation_context:
            raise ValueError("invocation_context is required and must be a dictionary")

        python_runtime_input = invocation_context.get("python_runtime_input", "")
        if isinstance(python_runtime_input, list):
            python_runtime_input = python_runtime_input[0]
        elif isinstance(python_runtime_input, str):
            pass
        else:
            raise ValueError("python_runtime_input must be a list or a string")
        runtime_globals = {}
        try:
            runtime_globals = {**globals()}
            exec(python_runtime_input, runtime_globals, None)
        except Exception as e:
            runtime_globals["final_result"] = str(e)
        return {"output": runtime_globals["final_result"]}, "output_1"

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        return {}, "output_1"
