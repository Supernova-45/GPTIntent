# modified from original Whisper Leak GPT-4o collector

import os
from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

from openai import OpenAI
import httpx
from dotenv import load_dotenv

class GPT5Mini(ChatbotBase):
    """
        OpenAI GPT 5 chatbot.
    """

    def __init__(self, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()

        # Validate environment variables
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError('OPENAI_API_KEY is not set in the environment variables.')

        # Create client that also saves the local port
        self._transport = LocalPortSaverTransport()
        self._client = OpenAI(
            base_url=f'https://api.openai.com/v1/',
            api_key=key,
            http_client=httpx.Client(transport=self._transport)
        )

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """

        # Send prompt
        response = []
        stream = self._client.chat.completions.create(
            extra_body={},
            model='gpt-5-mini',
            messages=[ { 'role': 'user', 'content': prompt } ],
            stream=True,
            temperature=temperature
        )
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                response.append(chunk.choices[0].delta.content)

        # Return response
        return (response, self._transport.get_local_port())

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 1.0
        return 1.0

    def get_common_name(self):
        """
            Gets the common name of the model.
        """

        # Return common name
        return 'gpt-5-mini'

    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """

        # Match
        return 'api.openai.com' in server_name