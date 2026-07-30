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
        return f"Task[{self.prompt_template.format(**self.prompt_params)} -> {self.answer}] \n\t-> Response: {self.response.text+'\n' if self.response else None} \n\tGrade: format={self.format_reward}, answer={self.answer_reward}, total={self.reward}"

    def __repr__(self) -> str:
        return self.__str__()

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
vllm_out: list[VLLMCompletion] = server.generate_completions([task.format_prompt() for task in tasks[:]], sample_param)
server.stop()

for i in range(len(vllm_out)):
    tasks[i].grade_response(vllm_out[i])
    # print(tasks[i])

# Accounting
fcac = [task for task in tasks if task.format_reward == 1 and task.answer_reward == 1]
fcai = [task for task in tasks if task.format_reward == 1 and task.answer_reward == 0]
fiac = [task for task in tasks if task.format_reward == 0 and task.answer_reward == 1]
fiai = [task for task in tasks if task.format_reward == 0 and task.answer_reward == 0]
print(f"FCAC={len(fcac)}, FCAI={len(fcai)}, FIAC={len(fiac)}, FIAI={len(fiai)}")

print(f"Question only fully correct: {len([task for task in fcac if task.prompt_template == question_only_template])}")
print(f"Question only format correct: {len([task for task in fcai if task.prompt_template == question_only_template])}")
print(f"Question only answer correct: {len([task for task in fiac if task.prompt_template == question_only_template])}")
print(f"Question only incorrect: {len([task for task in fiai if task.prompt_template == question_only_template])}")

print(f"R1 zero fully correct: {len([task for task in fcac if task.prompt_template == r1_zero_template])}")
print(f"R1 zero format correct: {len([task for task in fcai if task.prompt_template == r1_zero_template])}")
print(f"R1 zero answer correct: {len([task for task in fiac if task.prompt_template == r1_zero_template])}")
print(f"R1 zero incorrect: {len([task for task in fiai if task.prompt_template == r1_zero_template])}")

print(f"R1 zero three shot GSM8K fully correct: {len([task for task in fcac if task.prompt_template == r1_zero_three_shot_gsm8k_template])}")
print(f"R1 zero three shot GSM8K format correct: {len([task for task in fcai if task.prompt_template == r1_zero_three_shot_gsm8k_template])}")
print(f"R1 zero three shot GSM8K answer correct: {len([task for task in fiac if task.prompt_template == r1_zero_three_shot_gsm8k_template])}")
print(f"R1 zero three shot GSM8K incorrect: {len([task for task in fiai if task.prompt_template == r1_zero_three_shot_gsm8k_template])}")

with open("tasks.txt", "w") as f:
    f.write(f"Correct format \n\n\n {"\n".join([str(task) for task in fcai[:10]])}\n")
    f.write(f"Fully incorrect tasks \n\n\n {"\n".join([str(task) for task in fiai[:10]])}\n")

    f.write(f"All tasks: {"\n".join([str(task) for task in tasks])}")