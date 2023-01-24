from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple

from haystack import BaseComponent, Pipeline, MultiLabel, Document
from haystack.errors import PipelineConfigError
from haystack.nodes import PromptNode, PromptTemplate
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

    def __init__(
        self, prompt_node: PromptNode, tools: List[Dict[str, str]], tool_map: Optional[Dict[str, Pipeline]] = None
    ):

        """
        :param prompt_node: description
        :param tools: example {'pipeline_name': 'serpapi_pipeline', 'tool_name': 'Search', 'description': 'useful for when...'}
        """
        super().__init__()
        self.prompt_node = prompt_node
        self.tools = tools
        if tool_map is not None:
            self.tool_map = tool_map
        else:
            self.tool_map: Dict[str, Pipeline] = {}

    def run(self, query: str):
        tool_strings = "\n".join([f"{tool['tool_name']}: {tool['description']}" for tool in self.tools])
        tool_names = ", ".join([tool["tool_name"] for tool in self.tools])

        agent_scratchpad = ""
        prefix = """Answer the following questions as best as you can. You have access to the following tools:"""
        format_instructions = f"""Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
        suffix = f"""Begin!
Question: $query
Thought: {agent_scratchpad}"""
        # Question: {query}

        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])

        while True:
            # pred = self.prompt_node(text)
            prompt_template = PromptTemplate("t1", template, ["query"])
            pred = self.prompt_node.prompt(prompt_template=prompt_template, query=query)
            template += str(pred[0]) + "\n"
            action, action_input = self.get_action_and_input(llm_output=pred[0])
            if action == "Final Answer":
                return template, action_input  # TODO $query is not replaced
            next_pipeline = self.tool_map[action]
            result = next_pipeline.run(query=action_input)
            observation = result["output"]
            template += "Observation: " + str(observation) + "\n"

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
    ) -> "MRKLAgent":
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, one containing a MRKLAgent and all others as tools for that agent.

        :param path: Path to the YAML file to load
        :param pipeline_name: The name of the pipeline to load. That pipeline must contain a MRKLAgent.
        :param overwrite_with_env_variables: Overwrite the configuration with environment variables. For example, to change index name param for an ElasticsearchDocumentStore, an env variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an `_` sign must be used to specify nested hierarchical properties.
        :param strict_version_check: whether to fail in case of a version mismatch (throws a warning otherwise).
        """
        config = read_pipeline_config_from_yaml(path)
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

        tool_names = [tool["tool_name"] for tool in mrkl_agent.tools]
        tool_pipelines = [
            Pipeline.load_from_config(
                pipeline_config=config,
                pipeline_name=tool["pipeline_name"],
                overwrite_with_env_variables=overwrite_with_env_variables,
                strict_version_check=strict_version_check,
            )
            for tool in mrkl_agent.tools
            # The loaded YAML might contain more pipelines than we want to use in the MRKLAgent
            # Here, we collect only those pipelines that are explicitly specified in the MRKLAgent's tools parameter in the YAML
        ]

        mrkl_agent.tool_map = dict(zip(tool_names, tool_pipelines))
        return mrkl_agent

    @classmethod
    def load_from_config(
        cls,
        pipeline_config: Dict,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
    ) -> Pipeline:
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
        # validate_config(pipeline_config, strict_version_check=strict_version_check) # TODO reenable yaml validation
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

    def get_action_and_input(self, llm_output: str) -> Tuple[str, str]:
        """Parse out the action and input from the LLM output."""
        FINAL_ANSWER_ACTION = "Final Answer: "

        ps = [p for p in llm_output.split("\n") if p]
        if ps[-1].startswith("Final Answer"):
            directive = ps[-1][len(FINAL_ANSWER_ACTION) :]
            return "Final Answer", directive
        if not ps[-1].startswith("Action Input: "):
            raise ValueError("The last line does not have an action input, " "something has gone terribly wrong.")
        if not ps[-2].startswith("Action: "):
            raise ValueError("The second to last line does not have an action, " "something has gone terribly wrong.")
        action = ps[-2][len("Action: ") :]
        action_input = ps[-1][len("Action Input: ") :]
        return action, action_input.strip(" ").strip('"')
