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
    The MRKLAgent class answers queries by choosing between different actions/tools, which are implemented as pipelines.
    It uses a large language model (LLM) to generate a thought based on the query, choose an action/tool, and generate the input for the action/tool.
    Based on the result returned by an action/tool, the MRKLAgent has two options.
    It can either repeat the process of 1) thought, 2) action choice, 3) action input or it stops if it knows the answer.

    The MRKLAgent can be used for questions that contain multiple subquestions that can be answered step-by-step (Multihop QA).
    Combined with tools like Haystack's PythonRuntime or SerpAPIComponent, the MRKLAgent can query the web and do calculations.
    For example, the query "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?" can be broken down into three steps:
    1) Searching the web for the name of Olivia Wilde's boyfriend
    2) Searching the web for the age of that boyfriend
    3) Calculating that age raised to the 0.23 power

    The MRKLAgent can be either created programmatically or loaded from a YAML file.

    **Example programmatic creation:**
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

    **Example YAML file:**
        version: ignore
        components:
          - name: MRKLAgent
            type: MRKLAgent
            params:
              prompt_node: MRKLAgentPromptNode
              tools: [{'pipeline_name': 'serpapi_pipeline', 'tool_name': 'Search', 'description': 'useful for when you need to answer questions about current events. You should ask targeted questions'}, {'pipeline_name': 'calculator_pipeline', 'tool_name': 'Calculator', 'description': 'useful for when you need to answer questions about math'}]
          - name: MRKLAgentPromptNode
            type: PromptNode
            params:
              model_name_or_path: DavinciModel
              stop_words: ['Observation:']
          - name: DavinciModel
            type: PromptModel
            params:
              model_name_or_path: 'text-davinci-003'
              api_key: 'XYZ'
          - name: Serp
            type: SerpAPIComponent
            params:
              api_key: 'XYZ'
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

    outgoing_edges = 1

    def __init__(
        self, prompt_node: PromptNode, tools: List[Dict[str, str]], tool_map: Optional[Dict[str, Pipeline]] = None
    ):

        """
        :param prompt_node: The PromptNode that shall be used to generate thoughts, actions and action inputs
        :param tools: A description of the tools that the agent can use. For each tool, the pipeline name, the name of the tool/action and a description of what the tool is useful for is required. The expected format is a List of Dictionary. Example: {'pipeline_name': 'serpapi_pipeline', 'tool_name': 'Search', 'description': 'useful for when...'}
        :param tool_map: A map from name of a tool/action to the pipeline that implements it and can be called to run it. Optional parameter that is not needed in YAML files but only if a MRKLAgent is created programmatically.
        """
        super().__init__()
        self.prompt_node = prompt_node
        self.tools = tools
        if tool_map is not None:
            self.tool_map = tool_map
        else:
            self.tool_map: Dict[str, Pipeline] = {}

        tool_names = ", ".join([tool["tool_name"] for tool in self.tools])
        tool_names_with_description = "\n".join([f"{tool['tool_name']}: {tool['description']}" for tool in self.tools])

        prefix = "Answer the following questions as best as you can. You have access to the following tools:"
        format_instructions = (
            "Use the following format:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            f"Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question"
        )
        suffix = "Begin!\n" "Question: $query\n" "Thought: "
        self.template = "\n\n".join([prefix, tool_names_with_description, format_instructions, suffix])

    def run(self, query: str):
        # TODO align return format with BaseComponent.run(...) -> Tuple[Dict, str]
        notes = self.template
        while True:
            # pred = self.prompt_node(text)
            prompt_template = PromptTemplate("t1", notes, ["query"])
            pred = self.prompt_node.prompt(prompt_template=prompt_template, query=query)
            notes += str(pred[0]) + "\n"
            action, action_input = self.get_action_and_input(llm_output=pred[0])
            if action == "Final Answer":
                return notes, action_input  # TODO $query is not replaced
            next_pipeline = self.tool_map[action]
            result = next_pipeline.run(query=action_input)
            observation = result["output"]
            notes += "Observation: " + str(observation) + "\nThought: "

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
