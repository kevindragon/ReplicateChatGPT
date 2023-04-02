from dataclasses import dataclass
from typing import List
import openai


MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.5

openai.api_key = "YOUR OpenAI API key"
openai.proxy = "socks5h://127.0.0.1:1080"


@dataclass
class Prior:
    question: str
    answer: str


def chat(priors: List[Prior], question: str):
    messages = []
    for prior in priors:
        messages.append({"role": "user", "content": prior.question})
        messages.append({"role": "assistant", "content": prior.answer})
    messages.append({"role": "user", "content": question})
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True
    )
    return resp


class Session:
    def __init__(self):
        self.priors: List[Prior] = []

    def add_chat(self, question: str, answer: str):
        self.priors.append(Prior(question=question, answer=answer))

    def take_last_priors(self, n=2):
        return self.priors[-n:]


def main():
    session = Session()
    while True:
        question = input("问：")
        if question == "QUIT":
            print("Bye!", flush=True)
            exit()
        resp = chat(session.take_last_priors(2), question)
        answer = ""
        print("答：", end="", flush=True)
        for chunk in resp:
            chunk_message = chunk['choices'][0]['delta']
            content = chunk_message.get('content', '')
            answer += content
            print(content, end="", flush=True)
        print("\n"+"-"*10)
        session.add_chat(question, answer)


if __name__ == "__main__":
    main()
