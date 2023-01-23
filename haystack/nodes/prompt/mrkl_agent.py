import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

from haystack import BaseComponent, Pipeline, MultiLabel, Document
from haystack.errors import PipelineConfigError
from haystack.nodes import PromptNode
from haystack.pipelines.config import (
    read_pipeline_config_from_yaml,
    validate_config,
    get_pipeline_definition,
    get_component_definitions,
)


class MRKLAgent(BaseComponent):
    """
    Load tools as pipelines from YAML file.

    **Methods:**
    init() with list of pipelines, promptnode
    load_from_yaml()

    **Example:**
    self.tool_map: hashmap
    while
      use promptnode to select tool from action and action_input
      pass action_input into tool returned from map
      get observation
    """

    outgoing_edges = 1

    def __init__(self, pipeline_names: List[str], prompt_node: PromptNode):

        """
        :param prompt_node: description
        """
        super().__init__()
        self.tool_map: Dict[str, Pipeline] = {}  # map action to pipelines/pipeline_names
        self.prompt_node = prompt_node
        self.pipeline_names = pipeline_names

    def run(self, query: str):
        while True:
            pred, _ = self.prompt_node.run(query=query)
            action = pred["results"]
            action_input = None
            next_pipeline = self.tool_map[action]
            # TODO add action that breaks out of the loop
            result, _ = next_pipeline.run(query=action_input)
            observation = result["output"]
            query += observation

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
        raise NotImplementedError()

    @classmethod
    def load_from_yaml(
        cls,
        path: Path,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
    ):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.
        """
        config = read_pipeline_config_from_yaml(path)
        tool_pipeline_names = [p["name"] for p in config["pipelines"] if p["name"] != pipeline_name]
        tool_pipelines = [
            Pipeline.load_from_config(
                pipeline_config=config,
                pipeline_name=tool_pipeline_name,
                overwrite_with_env_variables=overwrite_with_env_variables,
                strict_version_check=strict_version_check,
            )
            for tool_pipeline_name in tool_pipeline_names
        ]
        mrkl_pipeline = Pipeline.load_from_config(
            pipeline_config=config,
            pipeline_name=pipeline_name,
            overwrite_with_env_variables=overwrite_with_env_variables,
            strict_version_check=strict_version_check,
        )
        mrkl_agent_nodes = [node for node in mrkl_pipeline.components.values() if isinstance(node, MRKLAgent)]
        if len(mrkl_agent_nodes) == 0:
            raise PipelineConfigError(
                f"The loaded pipeline {pipeline_name} contains no MRKLAgent node. Please use a pipeline that contains such a node if you want to load it as a MRKLAgent."
            )
        elif len(mrkl_agent_nodes) > 1:
            raise PipelineConfigError(
                f"The loaded pipeline {pipeline_name} contains more than one MRKLAgent node. Please use a pipeline that contains exactly one such node if you want to load it as a MRKLAgent."
            )

        mrkl_agent = mrkl_agent_nodes[0]

        # The loaded YAML might contain more pipelines than we want to use in the MRKLAgent
        # Add only those tool pipelines to the agent's tool map that are explicitly specified in the MRKLAgent's parameter in the YAML
        mrkl_agent.tool_map = {
            pn: p for (pn, p) in zip(tool_pipeline_names, tool_pipelines) if pn in mrkl_agent.pipeline_names
        }
        return mrkl_agent

    @classmethod
    def load_from_config(
        cls,
        pipeline_config: Dict,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
    ):
        """
        Load Pipeline from a config dict defining the individual components and how they're tied together to form
        a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        :param pipeline_config: the pipeline config as dict
        :param pipeline_name: if the config contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        :param strict_version_check: whether to fail in case of a version mismatch (throws a warning otherwise).
        """
        # validate_config(pipeline_config, strict_version_check=strict_version_check)
        pipeline = Pipeline()

        pipeline_definition = get_pipeline_definition(pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config, overwrite_with_env_variables=overwrite_with_env_variables
        )
        components: Dict[str, BaseComponent] = {}
        for node_config in pipeline_definition["nodes"]:
            component = Pipeline._load_or_get_component(
                name=node_config["name"], definitions=component_definitions, components=components
            )
            pipeline.add_node(component=component, name=node_config["name"], inputs=node_config["inputs"])

        return pipeline
