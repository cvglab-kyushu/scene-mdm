import openai
import tiktoken
import json
import os
import re
import argparse
import time
import glob
from tqdm import tqdm

enc = tiktoken.get_encoding("cl100k_base")


dir_system = os.path.join(os.path.dirname(__file__), 'system')
dir_query = os.path.join(os.path.dirname(__file__), 'query')

class ChatGPT:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = "gpt-3.5-turbo"

        self.messages = []
        self.max_token_length = 1000
        self.max_completion_length = 100
        self.last_response = None
        self.query = ''
        self.instruction = ''
        # load prompt file
        self.system_message = "<|im_start|>system\n"
        fp_system = os.path.join(dir_system, 'system.txt')
        with open(fp_system) as f:
            data = f.read()
        self.system_message += data
        self.system_message += "\n<|im_end|>\n"

        # load prompt file
        fp_query = os.path.join(dir_query, 'query.txt')
        with open(fp_query) as f:
            self.query = f.read()

    def reset_history(self):
        self.messages = []

    # See
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt#chatml
    def create_prompt(self):
        prompt = ""
        for message in self.messages:
            prompt += message['text']

        if len(enc.encode(prompt)) > self.max_token_length - \
                self.max_completion_length:
            print('prompt too long. truncated.')
            # truncate the prompt by removing the oldest two messages
            self.messages = self.messages[2:]
            prompt = self.create_prompt()
        return prompt

    def generate(self, sentence, is_user_feedback=False):

        self.messages = []
        if is_user_feedback:
            self.messages.append({'sender': 'user',
                                  'text': self.instruction})
        else:
            text_base = self.query
            if text_base.find('[SENTENCE]') != -1:
                text_base = text_base.replace('[SENTENCE]', sentence)
                self.instruction = text_base
            self.messages.append({'sender': 'user', 'text': text_base})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.create_prompt()
            }],
            temperature=0.1,
            max_tokens=self.max_completion_length,
            top_p=0.5,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<|im_end|>"])
        text = response['choices'][0]['message']['content']
        self.last_response = text


        if len(self.messages) > 0 and self.last_response is not None:
            self.messages.append(
                {"sender": "assistant", "text": self.last_response})
        return text


if __name__ == "__main__":

    sentence = "a person walks and sits on a chair."

    aimodel = ChatGPT()

    text = aimodel.generate(sentence, is_user_feedback=False)

    answer = text[text.find("Answer"):]

    print(answer)



