import together
import os
from dotenv import load_dotenv

load_dotenv()

together.api_key = os.getenv("TOGETHER_API_KEY")

model_list = together.Models.list()
model_name = "togethercomputer/RedPajama-INCITE-7B-Instruct"

output = together.Complete.create(
    prompt="<human>: What are Isaac Asimov's Three Laws of Robotics?\n<bot>:",
    model=model_name,
    max_tokens=256,
    temperature=0.8,
    top_k=60,
    top_p=0.6,
    repetition_penalty=1.1,
    stop=["<human>", "\n\n"],
)

# print generated text
print(output["prompt"][0] + output["output"]["choices"][0]["text"])
