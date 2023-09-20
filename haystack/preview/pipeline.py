from typing import Any, Dict
import logging
import json
import os
from collections import OrderedDict

from canals import Pipeline as CanalsPipeline
from canals.errors import PipelineRuntimeError

logger = logging.getLogger(__name__)


class Pipeline(CanalsPipeline):
    async def arun(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        self._clear_visits_count()
        self.warm_up()

        logger.info("Pipeline execution started.")
        inputs_buffer = OrderedDict(
            {
                node: {key: value for key, value in input_data.items() if value is not None}
                for node, input_data in data.items()
            }
        )
        pipeline_output: Dict[str, Dict[str, Any]] = {}

        if debug:
            logger.info("Debug mode ON.")
        self.debug = {}

        # *** PIPELINE EXECUTION LOOP ***
        # We select the nodes to run by popping them in FIFO order from the inputs buffer.
        step = 0
        while inputs_buffer:
            step += 1
            if debug:
                self._record_pipeline_step(step, inputs_buffer, pipeline_output)
            logger.debug("> Queue at step %s: %s", step, {k: list(v.keys()) for k, v in inputs_buffer.items()})

            component_name, inputs = inputs_buffer.popitem(last=False)  # FIFO

            # Make sure it didn't run too many times already
            self._check_max_loops(component_name)

            # **** IS IT MY TURN YET? ****
            # Check if the component should be run or not
            action = self._calculate_action(name=component_name, inputs=inputs, inputs_buffer=inputs_buffer)

            # This component is missing data: let's put it back in the queue and wait.
            if action == "wait":
                if not inputs_buffer:
                    # What if there are no components to wait for?
                    raise PipelineRuntimeError(
                        f"'{component_name}' is stuck waiting for input, but there are no other components to run. "
                        "This is likely a Canals bug. Open an issue at https://github.com/deepset-ai/canals."
                    )

                inputs_buffer[component_name] = inputs
                continue

            # This component did not receive the input it needs: it must be on a skipped branch. Let's not run it.
            if action == "skip":
                self.graph.nodes[component_name]["visits"] += 1
                inputs_buffer = self._skip_downstream_unvisited_nodes(
                    component_name=component_name, inputs_buffer=inputs_buffer
                )
                continue

            if action == "remove":
                # This component has no reason of being in the run queue and we need to remove it. For example, this can happen to components that are connected to skipped branches of the pipeline.
                continue

            # **** RUN THE NODE ****
            # It is our turn! The node is ready to run and all necessary inputs are present
            output = await self._arun_component(name=component_name, inputs=inputs)

            # **** PROCESS THE OUTPUT ****
            # The node run successfully. Let's store or distribute the output it produced, if it's valid.
            if not self.graph.out_edges(component_name):
                # Note: if a node outputs many times (like in loops), the output will be overwritten
                pipeline_output[component_name] = output
            else:
                inputs_buffer = self._route_output(
                    node_results=output, node_name=component_name, inputs_buffer=inputs_buffer
                )

        if debug:
            self._record_pipeline_step(step + 1, inputs_buffer, pipeline_output)

            # Save to json
            # NOTE: Ruff, rightly, doesn't like open in async functions
            # os.makedirs(self.debug_path, exist_ok=True)
            # with open(self.debug_path / "data.json", "w", encoding="utf-8") as datafile:
            #     json.dump(self.debug, datafile, indent=4, default=str)

            # Store in the output
            pipeline_output["_debug"] = self.debug  # type: ignore

        logger.info("Pipeline executed successfully.")
        return pipeline_output

    async def _arun_component(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Once we're confident this component is ready to run, run it and collect the output.
        """
        self.graph.nodes[name]["visits"] += 1
        instance = self.graph.nodes[name]["instance"]
        try:
            logger.info("* Running %s (visits: %s)", name, self.graph.nodes[name]["visits"])
            logger.debug("   '%s' inputs: %s", name, inputs)

            try:
                outputs = await instance.run(**inputs)
            except TypeError:
                outputs = instance.run(**inputs)

            # Unwrap the output
            logger.debug("   '%s' outputs: %s\n", name, outputs)

        except Exception as e:
            raise PipelineRuntimeError(
                f"{name} raised '{e.__class__.__name__}: {e}' \nInputs: {inputs}\n\n"
                "See the stacktrace above for more information."
            ) from e

        return outputs
