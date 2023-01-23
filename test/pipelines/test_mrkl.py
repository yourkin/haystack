import pytest

from haystack.nodes.prompt.mrkl_agent import MRKLAgent

from ..conftest import SAMPLES_PATH


@pytest.mark.integration
def test_load_agent():
    mrkl_agent = MRKLAgent.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "mrkl.haystack-pipeline.yml", pipeline_name="mrkl_query_pipeline"
    )
