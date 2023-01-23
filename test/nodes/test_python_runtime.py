import os

import pytest

from haystack import Pipeline


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_calculator_pipeline(tmp_path):
    api_key = os.environ.get("OPENAI_API_KEY", None)
    with open(tmp_path / "test_calculator_pipeline.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: pmodel_openai
              type: PromptModel
              params:
                model_name_or_path: text-davinci-003
                api_key: {api_key}
            - name: calculator_template
              type: PromptTemplate
              params:
                name: calculator
                prompt_text:  |
                    # Write a simple python function that calculates
                    # $query
                    # Do not print the result; invoke the function and assign the result to final_result variable
                    # Start with import statement
            - name: p1
              params:
                model_name_or_path: pmodel_openai
                default_prompt_template: calculator_template
                output_variable: python_runtime_input
              type: PromptNode
            - name: pr
              type: PythonRuntime
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
              - name: pr
                inputs:
                - p1
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "test_calculator_pipeline.yml")
    result = pipeline.run(query="2^3")
    assert result["output"] == 8.0
