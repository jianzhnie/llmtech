# System Prompt 和 Chat Template



##  在 Chat Template 中应用 System Prompt



```python
qwen_math_cot = ("Please reason step by step, and put your final answer within \\boxed{}.")

def chat_template_example(tokenizer: PreTrainedTokenizerBase) -> None:
    """
    Example usage of `apply_chat_template` for debugging purposes.
    """
    system_prompt = qwen_math_cot
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
        {"role": "assistant", "content": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n\\boxed{72}"}
    ]
    print(tokenizer.apply_chat_template(chat, tokenize=False))
```



```python
from transformers import AutoTokenizer

MODEL_PATHS: Dict[str, str] = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "QwQ-32B": "Qwen/QwQ-32B",
    "Qwen3-1.7B": "Qwen/Qwen3-14B",
    "AM-Thinking-v1": "a-m-team/AM-Thinking-v1",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Skywork-OR1-32B": "Skywork/Skywork-OR1-32B",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
}
for model_name, maodel_path in MODEL_PATHS.items():
	tokenizer = AutoTokenizer.from_pretrained(model_path)
  chat_template_example(tokenizer)
```



### Qwen/Qwen2.5-7B

```python
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{{}}.<|im_end|>
<|im_start|>user
'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
 \boxed{{72}}<|im_end|>
```



### Qwen/QwQ-32B

```python
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}<|im_end|>
```



### Qwen/Qwen3-14B

```python
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
<think>

</think>

Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}<|im_end|>
```



### a-m-team/AM-Thinking-v1



```python
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}<|im_end|>
```



### deepseek-ai/DeepSeek-R1



```python
<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \boxed{}.<｜User｜>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}<｜end▁of▁sentence｜>
```



###  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B



```python
<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \boxed{}.<｜User｜>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}<｜end▁of▁sentence｜>
```



### Skywork/Skywork-OR1-32B

```python
<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \boxed{}.<｜User｜>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}<｜end▁of▁sentence｜>
```



## 在 Chat Template 中应用 Math Cot  Prompt

```python
from transformers import AutoTokenizer

qwen_math_cot = ("Please reason step by step, and put your final answer within \\boxed{}.")


def chat_template_example(tokenizer: PreTrainedTokenizerBase) -> None:
    """
    Example usage of `apply_chat_template` for debugging purposes.
    """
    chat = [
        {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" +  "\n" +qwen_math_cot},
        {"role": "assistant", "content": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n\\boxed{72}"}
    ]
    print(tokenizer.apply_chat_template(chat, tokenize=False))


MODEL_PATHS: Dict[str, str] = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "QwQ-32B": "Qwen/QwQ-32B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "AM-Thinking-v1": "a-m-team/AM-Thinking-v1",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Skywork-OR1-32B": "Skywork/Skywork-OR1-32B",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
}
for model_name, maodel_path in MODEL_PATHS.items():
	tokenizer = AutoTokenizer.from_pretrained(model_path)
  chat_template_example(tokenizer)
```



### Qwen/Qwen2.5-7B

```python
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```



### Qwen/QwQ-32B

```python
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```



### Qwen/Qwen3-14B

```python
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
<think>

</think>

Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```



### a-m-team/AM-Thinking-v1

```python
<|im_start|>system
You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```



### deepseek-ai/DeepSeek-R1

```python
<｜begin▁of▁sentence｜><｜User｜>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```



### deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

```shell
<｜begin▁of▁sentence｜><｜User｜>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```



### Skywork/Skywork-OR1-32B

```python
<｜begin▁of▁sentence｜><｜User｜>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please reason step by step, and put your final answer within \boxed{}.<｜Assistant｜>Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
```
