import pytest
import os

from haystack import Pipeline
from haystack.nodes import SerpAPIComponent, PythonRuntime
from haystack.nodes.prompt.mrkl_agent import MRKLAgent

from ..conftest import SAMPLES_PATH


@pytest.mark.integration
def test_load_agent():
    mrkl_agent = MRKLAgent.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "mrkl.haystack-pipeline.yml", pipeline_name="mrkl_query_pipeline"
    )
    assert isinstance(mrkl_agent.tool_map["serpapi_pipeline"], Pipeline)
    assert isinstance(mrkl_agent.tool_map["serpapi_pipeline"].components["Serp"], SerpAPIComponent)

    assert isinstance(mrkl_agent.tool_map["calculator_pipeline"], Pipeline)
    assert isinstance(mrkl_agent.tool_map["calculator_pipeline"].components["Calculator"], PythonRuntime)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_run_agent(tmp_path):
    api_key = os.environ.get("OPENAI_API_KEY", None)
    with open(tmp_path / "test.mrkl.haystack-pipeline.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: MRKLAgent
                type: MRKLAgent
                params:
                  prompt_node: MRKLAgentPromptNode
                  pipeline_names: ['serpapi_pipeline', 'calculator_pipeline']
              - name: MRKLAgentPromptNode
                type: PromptNode
                params:
                  model_name_or_path: DavinciModel
              - name: DavinciModel
                type: PromptModel
                params:
                  model_name_or_path: 'text-davinci-003'
                  api_key: {api_key}
              - name: Serp
                type: SerpAPIComponent
                params:
                  api_key: 'XYZ'
              - name: CalculatorInput
                type: PromptNode
                params:
                  model_name_or_path: DavinciModel
                  default_prompt_template: CalculatorTemplate
              - name: Calculator
                type: PythonRuntime
              - name: CalculatorTemplate
                type: PromptTemplate
                params:
                  name: calculator
                  prompt_text:  |
                      # Write a simple python function that calculates
                      # $query
                      # Do not print the result; invoke the function and assign the result to final_result variable
                      # Start with import statement

            pipelines:
              - name: mrkl_query_pipeline
                nodes:
                  - name: MRKLAgent
                    inputs: [Query]
              - name: serpapi_pipeline
                nodes:
                  - name: Serp
                    inputs: [Query]
              - name: calculator_pipeline
                nodes:
                  - name: CalculatorInput
                    inputs: [Query]
                  - name: Calculator
                    inputs: [CalculatorInput]

        """
        )
    mrkl_agent = MRKLAgent.load_from_yaml(
        tmp_path / "test.mrkl.haystack-pipeline.yml", pipeline_name="mrkl_query_pipeline"
    )
    result = mrkl_agent.run(query="What is 2 to the power of 3?")
    assert result == "2 to the power of 3 is 8."
