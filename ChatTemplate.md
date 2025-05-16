# System Prompt 和 Chat Template 对训练和评估的影响

## 训练数据模板

### System Template && Chat Template

```python
from transformers import AutoTokenizer

system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."

model_name_or_path = 'Qwen/Qwen2.5-7B'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
chat = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'"},
    {"role": "assistant", "content": 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n \\boxed{{72}}'}]

result = tokenizer.apply_chat_template(chat, tokenize=False)
print(result)
```



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



### Qwen2.5  Chat Template

```python
from transformers import AutoTokenizer

model_name_or_path = 'Qwen/Qwen2.5-7B'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
chat = [
    {"role": "user", "content": "'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'"},
    {"role": "assistant", "content": 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n \\boxed{{72}}'}]

result = tokenizer.apply_chat_template(chat, tokenize=False)
print(result)
```



```python
You are a helpful assistant.<|im_end|>
<|im_start|>user
'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
 \boxed{{72}}<|im_end|>
```



## 评估模板

```python
qwen25-cot-template = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
    )
qwen25-chat-template = (
        "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )
default = ("{input}\n", "{output}", "\n\n\n")
```



##  训练评估结果对比

### 模型和数据

- Model： Qwen2.5-7B-Bae
- Train Dataset： GSM8k  Train
- Evaluation Dataset:   GSM8k  Test

### 训练曲线

![image-20250403101115391](../../../Library/Application%20Support/typora-user-images/image-20250403101115391.png)

### 结果

| Train                               | qwen25-cot-template | qwen25-chat-template | default |
| ----------------------------------- | ------------------- | -------------------- | ------- |
| System Prompt   & Chat Template     | 92.3                | 90.8                 | 89.8    |
| Chat Template                       | 92.6                | 91.7                 | 91.3    |
| No System Prompt,  No Chat Template | 64.4                | 89.8                 | 90.6    |

## 结论

训练和评估保持一致的模板和数据处理方式确保最佳模型性能。
