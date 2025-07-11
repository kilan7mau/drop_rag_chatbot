from .base_chunker import BaseChunker
from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
import backoff
from chunking import RecursiveTokenChunker
from utils import openai_token_count
import nltk
nltk.download('punkt')  
print(nltk.data.path)
print(nltk.__file__)

class LLMClient(ABC):
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        pass

class GeminiClient(LLMClient):
    def __init__(self, model_name="gemini-2.5-flash", api_key=None):
        super().__init__(model_name, api_key)
        self.client = genai.GenerativeModel(model_name)
        
    
    def parse_message(self, messages):
        mapping = {
            "user": "user",
            "assistant": "model"
        }
        return [
            {"role": mapping[mess["role"]], "parts": mess["content"]}
            for mess in messages
        ]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            messages = self.parse_message(messages)
            response = self.client.generate_content(
                [
                    {"role": "user", "parts": system_prompt},
                    {"role": "model", "parts": "I understand. I will strictly follow your instruction!"},
                    *messages
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e    
    
class OpenAIClient(LLMClient):
    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=api_key)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            gpt_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=gpt_messages,
                temperature=temperature
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e
            
    

class LLMAgenticChunker(BaseChunker):
    def __init__(self, organisation: str="google", api_key:str=None, model_name: str=None):
        if organisation == "openai":
            self.client = OpenAIClient(model_name, api_key=api_key)
        elif organisation == "google":
            genai.configure(api_key=api_key)
            self.client = GeminiClient(model_name)
        
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=openai_token_count
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in organizing educational content for e-learning. "
                    "The input text is from course materials, divided into chunks marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify points where the topic changes significantly, ensuring each section covers a coherent educational concept, lesson, or subtopic. "
                    "Consider transitions between concepts, definitions, examples, or sections of a lecture as potential split points. "
                    "Respond with a list of chunk IDs where splits should occur, in ascending order, with at least one split. "
                    "For example, if chunks 1 and 2 discuss 'Introduction to Python' but chunk 3 starts 'Data Structures', suggest a split after chunk 2. "
                    "Your response must be in the form: 'split_after: 3, 5' and only include chunk IDs >= {current_chunk}."
                )
            },
            {
                "role": "user",
                "content": (
                        "CHUNKED_TEXT: " + chunked_input + "\n\n"
                                                           "Respond only with the IDs of the chunks where a split should occur, in ascending order, >= " + str(
                    current_chunk) + "."
                        + (
                            f"\nThe previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages 
        
    def split_text(self, text):
        import re

        chunks = self.splitter.split_text(text)

        split_indices = []

        short_cut = len(split_indices) > 0

        from tqdm import tqdm

        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while True and not short_cut:
                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0

                chunked_input = ''

                for i in range(current_chunk, len(chunks)):
                    token_count += openai_token_count(chunks[i])
                    chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)
                while True:
                    result_string = self.client.create_message(messages[0]['content'], messages[1:], max_tokens=200, temperature=0.2)
                    # Use regular expression to find all numbers in the string
                    split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
                    numbers = re.findall(r'\d+', split_after_line)
                    # Convert the found numbers to integers
                    numbers = list(map(int, numbers))

                    # print(numbers)

                    # Check if the numbers are in ascending order and are equal to or larger than current_chunk
                    if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)):
                        break
                    else:
                        messages = self.get_prompt(chunked_input, current_chunk, numbers)
                        print("Response: ", result_string)
                        print("Invalid response. Please try again.")

                split_indices.extend(numbers)

                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()

        chunks_to_split_after = [i - 1 for i in split_indices]

        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs