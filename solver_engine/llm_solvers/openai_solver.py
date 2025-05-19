import os
from openai import OpenAI
from dotenv import load_dotenv

from solver_engine.llm_solvers.abc_llm_solver import LLMSolverBase
from solver_engine.llm_solvers.input_formatters import InputFormatter

load_dotenv()

class OpenAISolver(LLMSolverBase):
    """LLM Solver that uses the OpenAI API."""

    def __init__(
        self,
        formatter: InputFormatter,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        max_trials: int = 5,
    ):
        super().__init__(formatter, max_trials)
        self.model_name = model_name
        self._api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key not provided or found in OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self._api_key)

    @property
    def solver_name(self) -> str:
        return f"OpenAISolver-{self.model_name}"

    def _make_api_call(self, prompt: str) -> str:
        """Makes an API call to OpenAI."""
        try:
            # Using the Chat Completions API as it's the standard now
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert maze solver. Follow the output rules precisely."},
                    {"role": "user", "content": prompt}
                ],
            )
            # Extracting the text content from the response
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "" # Return empty if no content
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return ""