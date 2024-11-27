from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import pandas as pd

hf_repo = "Hvixze/llama3_2ep_wb"
hf_model = "unsloth.Q4_K_M.gguf"
dir = "models"
model_path = hf_hub_download(
    repo_id=hf_repo,
    filename=hf_model,
    cache_dir=dir

)
llm = Llama(model_path=model_path)
instruction = """
You are a moderator of questions on the marketplace.
You receive a customer's question and its description as input.
Your task is to evaluate whether the question is related to the characteristics of the product,
its capabilities, configuration, etc. or not.
If the input data contains a question about the delivery date, delivery delay, product defect,
promotions, discounts, product prices, refunds, the ability to buy, etc.,
or does not contain a question about the product at all, then
you need to output 0 .
If the question concerns its characteristics, size chart, material, capabilities and configuration of the product, etc.,
then you need to output 1.
The answer must contain one number - 0 or 1.
"""

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def predict(row: pd.Series):
    prompt_string = alpaca_prompt.format(
    instruction,
    row["Question"] + "/n" + row["desc"],
    ""
)

    # Генерация текста
    output = llm(prompt_string, max_tokens=128, temperature=0)

    # Извлечение ответа
    return output['choices'][0]['text']

def answer(q: str):

    
    return "some"


if __name__ == "__main__":
    pass