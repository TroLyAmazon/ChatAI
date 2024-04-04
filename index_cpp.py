from llama_cpp import Llama

# conda activate llama
# https://github.com/abetlen/llama-cpp-python

llm = Llama(model_path="./models/mistral-7b-openorca.Q8_0.gguf",
            n_gpu_layers=1, n_ctx=4096)

output = llm.create_completion("""<|im_start|>system
You are a helpful chatbot.
<|im_end|>
<|im_start|>user
Miêu tả về bạn?<|im_end|>
<|im_start|>assistant""", max_tokens=1000,  stop=["<|im_end|>"], stream=True)

for token in output:
    print(token["choices"][0]["text"], end='', flush=True)
