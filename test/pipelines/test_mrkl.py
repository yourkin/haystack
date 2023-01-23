import pytest

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


@pytest.mark.skip(reason="MRKLAgent does not support run() yet.")
@pytest.mark.integration
def test_run_agent():
    mrkl_agent = MRKLAgent.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "mrkl.haystack-pipeline.yml", pipeline_name="mrkl_query_pipeline"
    )
    mrkl_agent.run(query="What is 2 to the power of 3?")
