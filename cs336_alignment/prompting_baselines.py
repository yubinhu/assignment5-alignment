from dataclasses import dataclass
from typing import Callable
import vllm_utils
from vllm_utils import VLLMCompletion

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

@dataclass
class Task:
    prompt_params: dict
    prompt_template: str
    reward_fn: Callable
    answer: str
    response: VLLMCompletion | None = None
    format_reward: float | None = None
    answer_reward: float | None = None
    reward: float | None = None

    def format_prompt(self) -> str:
        return self.prompt_template.format(**self.prompt_params)

    def grade_response(self, response: VLLMCompletion) -> None:
        self.response = response
        rewards = self.reward_fn(response.text, self.answer)
        self.format_reward = rewards["format_reward"]
        self.answer_reward = rewards["answer_reward"]
        self.reward = rewards["reward"]

    def __str__(self) -> str:
        return f"Task[{self.prompt_template.format(**self.prompt_params)[:50]} -> {self.answer[:50]}] -> Response: {self.response.text[:50]} -> Format: {self.format_reward}, Answer: {self.answer_reward}, Total: {self.reward}"

server = vllm_utils.VLLMServer("allenai/OLMo-2-0425-1B", gpu=0)
sample_param = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 41
}

with open("cs336_alignment/prompts/question_only.prompt", "r") as f:
    question_only_template = f.read()

with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
    r1_zero_template = f.read()

with open("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt", "r") as f:
    r1_zero_three_shot_gsm8k_template = f.read()

tasks: list[Task] = []
import json
with open("data/gsm8k/test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        question = {"question": data["question"]}
        answer = data["answer"].split("####")[-1].strip()
        tasks.append(Task(
            prompt_params=question,
            prompt_template=question_only_template,
            reward_fn=question_only_reward_fn,
            answer=answer,
        ))
        tasks.append(Task(
            prompt_params=question,
            prompt_template=r1_zero_template,
            reward_fn=r1_zero_reward_fn,
            answer=answer,
        ))
        tasks.append(Task(
            prompt_params=question,
            prompt_template=r1_zero_three_shot_gsm8k_template,
            reward_fn=r1_zero_reward_fn,
            answer=answer,
        ))

server.start()
vllm_out: list[VLLMCompletion] = server.generate_completions([task.format_prompt() for task in tasks[:10]], sample_param)
for i in range(len(vllm_out)):
    tasks[i].grade_response(vllm_out[i])
    print(tasks[i])
server.stop()