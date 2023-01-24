import os

import pytest

from haystack import Pipeline
from haystack.nodes.other.serpapi import SerpAPIComponent


@pytest.mark.skipif(
    not os.environ.get("SERPAPI_API_KEY", None),
    reason="Please export an env var called SERPAPI_API_KEY containing the SERP API key to run this test.",
)
def test_serp_api():
    search = SerpAPIComponent(api_key=os.environ.get("SERPAPI_API_KEY"))
    result, _ = search.run(query="Olivia Wilde boyfriend")
    assert result["output"] in ["Jason Sudeikis", "Harry Styles"]


@pytest.mark.skipif(
    not os.environ.get("SERPAPI_API_KEY", None),
    reason="Please export an env var called SERPAPI_API_KEY containing the SERP API key to run this test.",
)
def test_serpapi_pipeline(tmp_path):
    api_key = os.environ.get("SERPAPI_API_KEY", None)
    with open(tmp_path / "test_serpapi_pipeline.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: serp
              params:
                api_key: {api_key}
              type: SerpAPIComponent
            pipelines:
            - name: query
              nodes:
              - name: serp
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "test_serpapi_pipeline.yml")
    result = pipeline.run(query="Olivia Wilde boyfriend")
    assert result["output"] in ["Jason Sudeikis", "Harry Styles"]
