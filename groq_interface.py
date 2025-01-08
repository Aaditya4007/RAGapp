import os
from groq import Groq
from typing import Optional

class GroqException(Exception):
    """
    Custom exception class for Groq API errors.
    """
    def __init__(self, error: dict):
        self.error = error
        super().__init__(error.get('message', 'Unknown Groq API error'))

class GroqLLMInterface:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        """
        Initialize Groq LLM Interface
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API Key not found. Please provide it.")

        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate_response(
        self,
        system_message: str,
        user_message: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> str:
        """
        Generate a response using Groq LLM
        """
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                if hasattr(e, 'error') and e.error.get('type') == 'rate_limit_exceeded':
                    retries += 1
                    print(f"Rate limit exceeded, retrying... (attempt {retries}/{max_retries})")
                    continue
                else:
                    raise GroqException(e.error) from e
        raise RuntimeError("Maximum number of retries reached. Unable to generate response.")