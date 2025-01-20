import ujson as json
import time

from openai import OpenAI, BadRequestError
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential


class MinimumDelay:
    def __init__(self, delay: float | int):
        self.delay = delay
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        seconds = end - self.start
        if self.delay > seconds:
            time.sleep(self.delay - seconds)


@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(3))
def chat(client: OpenAI, delay: float | int, **kwargs) -> ChatCompletion | None:
    try:
        with MinimumDelay(delay):
            return client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        print(f"Bad Request: {e}")
        if "safety" in e.message:
            return None
        raise e
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def print_messages(messages, max_size: int = 1000):
    for message in messages:
        print(f"{message['role']}:")
        if message["role"] == "assistant":
            json_message = json.loads(message["content"])
            print(json.dumps(json_message, indent=2)[:max_size] + "...")
        else:
            print(message["content"][:max_size] + "...")
        print()
        print()
