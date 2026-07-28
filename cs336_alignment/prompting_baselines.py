import vllm_utils
from vllm_utils import VLLMCompletion

server = vllm_utils.VLLMServer("allenai/OLMo-2-0425-1B", gpu=0)
sample_param = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 41
}

server.start()
vllm_out: list[VLLMCompletion] = server.generate_completions(["Good morning! "], sample_param)
print(vllm_out[0].text)

server.stop()