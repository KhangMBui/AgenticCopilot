"""
Manual ReAct agent implementation.

ReAct = "Reason + Act" (interleaved)

Instead of:
- LLM answers directly in one shot

ReAct does:
1) Thought  -> LLM decides what to do next
2) Action   -> choose a tool (search, retrieve, calculator, etc.)
3) Observation -> tool returns output
4) Repeat until a final answer is produced or we hit max_steps

Key idea:
- The LLM is the "planner/brain"
- Tools are the "hands"
- Observation is the "feedback loop" that helps the LLM adjust.
"""

import json

from core.agents.ReAct.state import AgentState, AgentStep
from core.agents.ReAct.parser import parse_tool_call, extract_thought
from core.tools import ToolRegistry
from core.llm import LLMClient, Message


class ReActAgent:
    """
    ReAct agent: Reasoning + Acting in an interleaved manner.

    High-level loop:
    - We maintain an AgentState (query + history of steps).
    - Each step:
        a) build a prompt that includes:
           - the user query
           - available tools and how to call them
           - full history so far (Thought/Action/Observation)
        b) ask the LLM to produce the *next* Thought + Action (+ Action Input)
        c) parse the tool call from model output
        d) execute the tool through ToolRegistry
        e) store observation
        f) repeat

    Termination:
    - When Action == "final_answer"
    - or when max_steps is exceeded
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        # LLMClient is our abstraction over the model provider
        # (OpenAI, local model, etc.)
        self.llm = llm

        # ToolRegistry holds all registered tools and can execute them by name.
        self.tools = tools

        # Safety/cost control: prevents infinite loops and runaway tool calls.
        self.max_steps = max_steps

        # If True, prints each step (useful for debugging/demo)
        self.verbose = verbose

    def run(self, query: str) -> AgentState:
        """
        Execute the ReAct loop end-to-end.

        Args:
            query: User's question

        Returns:
            AgentState containing:
              - query
              - list of AgentSteps (history)
              - status (completed / max_steps / error)
              - final_answer (if completed)
        """

        # Initialize state with just the query and empty history.
        state = AgentState(query=query)

        # Continue until:
        # - state signals finished
        # - OR we exceed max steps
        #
        # NOTE: off-by-one details depend on your AgentState.current_step_number().
        # If it starts at 1, use <= max_steps. If it starts at 0, adjust.
        while not state.is_finished() and state.current_step_number() <= self.max_steps:
            # Execute one iteration of ReAct (Think -> Act -> Observe)
            step = self._execute_step(state)

            # Record step into state history
            state.add_step(step)

            # Optional debug printing
            if self.verbose:
                self._print_step(step)

            # If model chose "final_answer" tool, we finish the loop.
            # In this code, step.observation is used as the final answer.
            if step.is_final:
                state.status = "completed"
                state.final_answer = step.observation
                break

        # If we exit because we hit max steps, mark status accordingly.
        if state.current_step_number() > self.max_steps:
            state.status = "max_steps"
            state.error = f"Reached maximum steps ({self.max_steps})"

        return state

    def _execute_step(self, state: AgentState) -> AgentStep:
        """
        Execute a single ReAct step.

        Step algorithm:
        1) Build the prompt (tools + history + query)
        2) Call LLM to produce next 'Thought', 'Action', 'Action Input'
        3) Parse Thought + Action + params from LLM output
        4) Execute the tool via ToolRegistry
        5) Return an AgentStep with the outcome
        """
        step_num = state.current_step_number()

        # ---------------------------------------------------------------------
        # 1) Build the ReAct prompt
        # ---------------------------------------------------------------------
        prompt = self._build_prompt(state)

        # 2) Ask the LLM for the next step
        # ---------------------------------------------------------------------
        # We wrap prompt as a single "user" message.
        # Many ReAct setups use a system prompt + user prompt; we can add later.
        messages = [Message(role="user", content=prompt)]

        # temperature=0.0 -> more deterministic (good for tool calling)
        response = self.llm.generate(messages, temperature=0.0, max_tokens=500)

        # The LLM output should follow the format:
        # Thought: ...
        # Action: tool_name
        # Action Input: {...json...}
        llm_output = response.content

        # ---------------------------------------------------------------------
        # 3) Extract the Thought portion (for logs / transparency)
        # ---------------------------------------------------------------------
        thought = extract_thought(llm_output)

        # ---------------------------------------------------------------------
        # 4) Parse tool call (Action + Action Input)
        # ---------------------------------------------------------------------
        tool_name, params, parse_error = parse_tool_call(llm_output)

        # If parsing fails, we return an error step. The loop will continue,
        # but we might want to consider stopping on repeated parse failures.
        if parse_error:
            return AgentStep(
                step_number=step_num,
                thought=thought,
                observation=f"Error parsing tool call: {parse_error}",
                success=False
            )

        # If the model produced no Action line, we produce an observation telling it
        # to follow the format. This is a "self-correction" hint.
        if not tool_name:
            return AgentStep(
                step_number=step_num,
                thought=thought,
                observation="No action specified. Please specify an Action and Action Input.",
                success=False
            )

        # ---------------------------------------------------------------------
        # 5) Execute the tool
        # ---------------------------------------------------------------------
        # ToolRegistry.execute returns something like:
        #   ToolResult(success: bool, output: str, error: str)
        #
        # If the tool fails, we feed the error back as the observation so the LLM
        # can adjust.
        result = self.tools.execute(tool_name, params)

        # ReAct termination convention:
        # - If the agent chooses Action: final_answer
        #   we stop and return the "answer"
        is_final = tool_name == "final_answer"

        return AgentStep(
            step_number=step_num,
            thought=thought,
            action=tool_name,
            action_input=params,
            observation=result.output if result.success else result.error,
            is_final=is_final,
            success=result.success
        )

    def _build_prompt(self, state: AgentState) -> str:
        """
        Build the ReAct prompt that will be sent to the LLM.

        It includes:
        - The user's original question
        - A list of tools (names + descriptions + parameter specs)
        - The history of previous Thought/Action/Observation steps
        - The required output format (so parsing is possible)

        Why we include history:
        - The model needs to see what it already tried and what happened.
        - The observation is how the model learns "what the tool returned".
        """

        # ---------------------------------------------------------------------
        # Build a readable "tools list" for the prompt
        # ---------------------------------------------------------------------
        tool_descriptions = []
        for schema in self.tools.get_schemas():
            # For each tool, list its parameters and descriptions
            params_desc = ", ".join(
                [f"{p.name} ({p.type}): {p.description}" for p in schema.parameters]
            )
            tool_descriptions.append(
                f"- {schema.name}: {schema.description}\n  Parameters: {params_desc}"
            )

        tools_text = "\n".join(tool_descriptions)

        # ---------------------------------------------------------------------
        # Build "history" text from prior steps
        # ---------------------------------------------------------------------
        history_text = ""
        for step in state.steps:
            history_text += f"\nThought: {step.thought}\n"
            if step.action:
                history_text += f"Action: {step.action}\n"
                history_text += f"Action Input: {json.dumps(step.action_input)}\n"
                history_text += f"Observation: {step.observation}\n"

        # ---------------------------------------------------------------------
        # Final prompt text the LLM sees
        # ---------------------------------------------------------------------
        prompt = f"""Answer the following question using the available tools.
                    Question: {state.query}

                    Available Tools:
                    {tools_text}

                    Use EXACTLY this format (no extra text after Action Input):

                    Thought: reasoning about what to do next
                    Action: tool_name
                    Action Input: {{"param": "value"}}

                    STOP after Action Input. Do not write "Observation" - I will provide it.

                    After I provide the Observation, continue with the next Thought/Action or use final_answer when ready.

                    When you have the final answer, use:
                    Action: final_answer
                    Action Input: {{"answer": "your complete answer"}}

                    Begin!
                    {history_text}
                    Thought:"""
        return prompt

    def _print_step(self, step: AgentStep):
        """
        Print one step of execution.

        Useful for:
        - debugging
        - demo videos
        - understanding where the agent is failing (parsing vs tool errors)
        """
        print(f"\n=== Step {step.step_number} ===")
        print(f"Thought: {step.thought}")
        if step.action:
            print(f"Action: {step.action}")
            print(f"Input: {step.action_input}")
            print(f"Observation: {step.observation}")
