from cs336_alignment import vllm_utils
from cs336_alignment.vllm_utils import VLLMCompletion
from cs336_alignment.solution import Task, load_gsm8k_examples
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn


def format_result(task: Task, response: VLLMCompletion, scores: dict[str, float]) -> str:
    return (
        f"Task[{task.format_prompt()} -> {task.answer}]\n"
        f"\t-> Response: {response.text}\n"
        f"\tGrade: format={scores['format_reward']}, "
        f"answer={scores['answer_reward']}, total={scores['reward']}"
    )


def main() -> None:
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
    for question_text, answer in load_gsm8k_examples("data/gsm8k/test.jsonl"):
        question = {"question": question_text}
        tasks.append(Task(
            prompt_params=question,
            prompt_template=question_only_template,
            answer=answer,
        ))
        tasks.append(Task(
            prompt_params=question,
            prompt_template=r1_zero_template,
            answer=answer,
        ))
        tasks.append(Task(
            prompt_params=question,
            prompt_template=r1_zero_three_shot_gsm8k_template,
            answer=answer,
        ))

    server.start()
    vllm_out: list[VLLMCompletion] = server.generate_completions([task.format_prompt() for task in tasks[:]], sample_param)
    server.stop()

    reward_fns = {
        question_only_template: question_only_reward_fn,
        r1_zero_template: r1_zero_reward_fn,
        r1_zero_three_shot_gsm8k_template: r1_zero_reward_fn,
    }
    # Keep each task, completion, and score dictionary together for reporting.
    results = [
        (task, response, reward_fns[task.prompt_template](response.text, task.answer))
        for task, response in zip(tasks, vllm_out, strict=True)
    ]
    categories = {
        (fmt, ans): [result for result in results
                     if result[2]["format_reward"] == fmt
                     and result[2]["answer_reward"] == ans]
        for fmt, ans in [(1, 1), (1, 0), (0, 1), (0, 0)]
    }
    fcac, fcai, fiac, fiai = (categories[key] for key in [(1, 1), (1, 0), (0, 1), (0, 0)])
    print(f"FCAC={len(fcac)}, FCAI={len(fcai)}, FIAC={len(fiac)}, FIAI={len(fiai)}")

    for name, template in [
        ("Question only", question_only_template),
        ("R1 zero", r1_zero_template),
        ("R1 zero three shot GSM8K", r1_zero_three_shot_gsm8k_template),
    ]:
        for label, group in [
            ("fully correct", fcac), ("format correct", fcai),
            ("answer correct", fiac), ("incorrect", fiai),
        ]:
            count = sum(task.prompt_template == template for task, _, _ in group)
            print(f"{name} {label}: {count}")

    with open("tasks.txt", "w") as f:
        f.write("Correct format \n\n\n " + "\n".join(
            format_result(*result) for result in fcai[:10]) + "\n")
        f.write("Fully incorrect tasks \n\n\n " + "\n".join(
            format_result(*result) for result in fiai[:10]) + "\n")
        f.write("All tasks: \n\n\n " + "\n".join(
            format_result(*result) for result in results))


if __name__ == "__main__":
    main()
