import pytest
import os

from haystack import Pipeline
from haystack.nodes import SerpAPIComponent, PythonRuntime, PromptNode, PromptModel
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
    not os.environ.get("OPENAI_API_KEY", None) or not os.environ.get("SERPAPI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key and an env var called SERPAPI_API_KEY containing the SERP API key to run this test.",
)
@pytest.mark.integration
def test_run_agent(tmp_path):
    openai_api_key = os.environ.get("OPENAI_API_KEY", None)
    serp_api_key = os.environ.get("SERPAPI_API_KEY", None)
    with open(tmp_path / "test.mrkl.haystack-pipeline.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: MRKLAgent
                type: MRKLAgent
                params:
                  prompt_node: MRKLAgentPromptNode
                  tools: [{{'pipeline_name': 'serpapi_pipeline', 'tool_name': 'Search', 'description': 'useful for when you need to answer questions about current events. You should ask targeted questions'}}, {{'pipeline_name': 'calculator_pipeline', 'tool_name': 'Calculator', 'description': 'useful for when you need to answer questions about math'}}]
              - name: MRKLAgentPromptNode
                type: PromptNode
                params:
                  model_name_or_path: DavinciModel
                  stop_words: ['Observation:']
              - name: DavinciModel
                type: PromptModel
                params:
                  model_name_or_path: 'text-davinci-003'
                  api_key: {openai_api_key}
              - name: Serp
                type: SerpAPIComponent
                params:
                  api_key: {serp_api_key}
              - name: CalculatorInput
                type: PromptNode
                params:
                  model_name_or_path: DavinciModel
                  default_prompt_template: CalculatorTemplate
                  output_variable: python_runtime_input
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
    assert result[1] == "2 to the power of 3 is 8."

    # result = mrkl_agent.run(query="Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
    # assert "Harry Styles" in result[1] and "2.15" in result[1]


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None) or not os.environ.get("SERPAPI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key and an env var called SERPAPI_API_KEY containing the SERP API key to run this test.",
)
@pytest.mark.integration
def test_run_agent_programmatically():
    prompt_model = PromptModel(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))
    prompt_node = PromptNode(model_name_or_path=prompt_model, stop_words=["Observation:"])

    search = SerpAPIComponent(api_key=os.environ.get("SERPAPI_API_KEY"))
    search_pipeline = Pipeline()
    search_pipeline.add_node(component=search, name="Serp", inputs=["Query"])

    tools = [
        {
            "pipeline_name": "serpapi_pipeline",
            "tool_name": "Search",
            "description": "useful for when you need to answer questions about current events. You should ask targeted questions",
        }
    ]
    tool_map = {"Search": search_pipeline}
    mrkl_agent = MRKLAgent(prompt_node=prompt_node, tools=tools, tool_map=tool_map)

    result = mrkl_agent.run(query="What is 2 to the power of 3?")
    assert "8" in result[1]
