# 文本大语言模型的聊天模板入门指南

大语言模型（LLMs）日益常见的应用场景是**聊天对话**。在聊天场景中，模型不再是延续单个文本字符串（如标准语言模型那样），而是延续由多条**消息**组成的对话。每条消息都包含一个**角色**（如"user"或"assistant"）和消息内容。

与分词处理类似，不同模型对聊天输入的格式要求差异很大。为此我们引入了**聊天模板**功能。聊天模板是纯文本LLMs分词器或多模态LLMs处理器的组成部分，它指定了如何将对话（表示为消息列表）转换为符合模型预期的单个可分词字符串。

本页将重点介绍纯文本LLMs的聊天模板基础用法。关于多模态模型的详细指南，我们准备了专门的[多模态模型文档](./chat_template_multimodal)，涵盖如何在模板中处理图像、视频和音频输入。

让我们通过`mistralai/Mistral-7B-Instruct-v0.1`模型的示例来具体说明：

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

注意分词器如何添加控制标记[INST]和[/INST]来标识用户消息的起止（但不用于assistant消息），并将整个对话合并为单个字符串。若使用默认的`tokenize=True`，该字符串还会被自动分词。

现在尝试将模型替换为`HuggingFaceH4/zephyr-7b-beta`，会得到：

```python
<|user|>
Hello, how are you?</s>
<|assistant|>
I'm doing great. How can I help you today?</s>
<|user|>
I'd like to show off how chat templating works!</s>
```

Zephyr和Mistral-Instruct都基于同一基础模型`Mistral-7B-v0.1`微调，但使用了完全不同的聊天格式。没有聊天模板时，您需要为每个模型编写手动格式化代码，而细微的格式错误就可能影响性能！聊天模板帮您处理格式化细节，让您可以编写适用于任何模型的通用代码。

## 如何使用聊天模板？

如示例所示，使用聊天模板非常简单。只需构建包含`role`和`content`键的消息列表，然后根据模型类型调用分词器的[`~PreTrainedTokenizer.apply_chat_template`]方法或处理器的[`~ProcessorMixin.apply_chat_template`]方法。当使用聊天模板作为模型生成输入时，建议设置`add_generation_prompt=True`来添加[生成提示](#什么是生成提示)。

### **纯文本LLMs与聊天模板的适配**

以下是使用Zephyr准备模型生成输入的示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # 建议使用bfloat16精度并转移到GPU

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```
输出符合Zephyr预期的格式：
```python
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

现在输入已正确格式化，可以使用模型生成响应：

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

输出示例：
```python
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

看，使用模板后生成响应变得非常简单！

### **多模态 LLMs与聊天模板的适配**

对于像 [LLaVA](<url id="" type="url" status="" title="" wc="">https://huggingface.co/llava-hf</url> ) 这样的多模态 LLMs，提示可以以类似的方式格式化。唯一的区别是你需要同时传递输入图像/视频以及文本。每个 `"content"` 必须是一个包含文本或图像/视频的列表。

以下是使用 `LLaVA` 模型准备输入的示例：

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)  # You may want to use bfloat16 and/or move to GPU here
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
          {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
          {"type": "text", "text": "What are these?"},
        ],
    },
]

processed_chat = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
print(processor.batch_decode(processed_chat["input_ids"][:, :30]))
```

该代码会生成符合LLaVA预期输入格式的字符串，末尾包含多个<image>标记。这些<image>标记是占位符，在前向传播时会被实际图像嵌入替换。`processed_chat` 可以进一步传递给 [generate()](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate</url> ) 以生成文本。

```python
'<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|><|im_start|>user <image><image><image><image><image><image><image><image>'
```

啊，原来这么简单！

## 是否有自动化的聊天流程？

是的！我们的text generation pipelines支持聊天输入，可以轻松使用聊天模型。过去我们使用专用的"ConversationalPipeline"类，现其功能已合并到[`TextGenerationPipeline`]。用pipeline重试Zephyr示例：

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # 打印assistant回复
```

输出：
```python
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```

pipeline 会自动处理分词和模板应用，只要模型有聊天模板，初始化pipeline后直接传入消息列表即可！

## 什么是"生成提示"？

你可能已经注意到 `apply_chat_template` 方法有一个 `add_generation_prompt` 参数。此参数告诉模板添加表示助手响应开始的标记。例如，考虑以下聊天：

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

不使用生成提示时的ChatML格式：
```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

使用生成提示时：
```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

需要注意的是，这次我们添加了指示模型开始生成回复的特殊标记。这种做法能确保模型在生成文本时，会按照预期输出机器人回复，而不是产生意外行为（例如延续用户消息）。请记住，聊天模型本质上仍是语言模型——它们经过训练是为了延续文本，而对话对它们而言只是特殊形式的文本！必须通过恰当的控制标记进行引导，才能使模型明确当前的任务目标。

并非所有模型都需要生成提示。部分模型（如LLaMA）在机器人回复前没有特殊标记。对于这类模型，`add_generation_prompt` 参数将不会产生任何效果。该参数的具体作用效果取决于实际使用的模板配置。


## "continue_final_message"的作用？

当使用`apply_chat_template`或`TextGenerationPipeline`时，设置`continue_final_message=True`可以让模型延续最后一条消息而不是开始新回复。这是通过移除最后消息的结束标记实现的，适用于"预填充"模型响应：

```python
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```

该模型将生成延续现有JSON字符串的文本，而非创建新消息。当您明确知道希望模型如何开始回复时，这种方法能有效提升模型遵循指令的准确性。

注意`add_generation_prompt`和`continue_final_message`不能同时使用。原因是`add_generation_prompt`和`continue_final_message`在输出结构上存在冲突：
1. **`add_generation_prompt`**：通过添加新消息起始标记（如`"助手："`），强制模型**开启全新回复**，适用于对话式交互场景。
2. **`continue_final_message`**：通过移除消息结束标记（如`</s>`），强制模型**延续当前内容流**，避免生成封闭式结尾。

由于前者要求"新开对话"，后者要求"延续对话"，二者逻辑互斥，同时使用将引发矛盾导致报错。

`TextGenerationPipeline`默认设置`add_generation_prompt=True`。但如果输入的最后消息是 assistant 角色，会自动切换为`continue_final_message=True`，因为多数模型不支持连续 assistant 消息。可通过显式传递参数覆盖此行为。

## 可以在训练中使用聊天模板吗？

是的！这是确保聊天模板与模型训练时所见标记一致的有效方法。我们建议您将聊天模板作为数据集的预处理步骤应用。完成后，您可以像处理其他语言模型训练任务一样继续操作。在训练过程中，通常应设置`add_generation_prompt=False`，因为用于触发助手回复的附加标记在训练阶段并无益处。来看个示例：

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```
输出：
```python
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

在此之后，您只需像处理标准语言模型训练任务一样继续训练流程，使用`formatted_chat`列作为输入即可。

**注意**：默认情况下，部分分词器会对文本自动添加特殊标记（如`<bos>`起始符和`<eos>`结束符）。由于聊天模板已包含所需的所有特殊标记，额外添加的标记可能导致重复或冲突，进而损害模型性能。因此：

1. **若通过`apply_chat_template(tokenize=False)`格式化文本**：
   后续分词时需显式设置`add_special_tokens=False`以避免重复添加标记。

   ```python
   text = apply_chat_template(..., tokenize=False)
   tokenized = tokenizer(text, add_special_tokens=False)  # 关闭自动添加特殊标记
   ```

2. **若使用`apply_chat_template(tokenize=True)`**：
   模板已集成分词逻辑，无需额外处理，系统会自动规避重复标记问题。




## 高级：聊天模板的额外输入

`apply_chat_template` 方法唯一需要的参数是 `messages`。然而，你可以将任何关键字参数传递给 `apply_chat_template`，它将在模板中可用。这为你提供了很大的自由度，可以将聊天模板用于许多事情。这些参数的名称或格式没有限制——你可以传递字符串、列表、字典或任何其他内容。

尽管如此，这些额外参数有一些常见的用例，例如传递函数调用的工具或检索增强生成的文档。在这些常见情况下，我们对这些参数的名称和格式有一些建议，这些建议在下面的章节中描述。我们鼓励模型作者使他们的聊天模板与这种格式兼容，以便在模型之间轻松转移工具调用代码。

## 高级：工具使用/函数调用

“工具使用” LLMs 可以选择在生成答案之前调用外部工具。当向工具使用模型传递工具时，只需将函数列表传递给 `tools` 参数：

```python
import datetime

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```

为了使这正常工作，你应该按照以下格式编写函数，以便它们可以被正确解析为工具：

- 函数应具有描述性的名称
- 每个参数必须具有类型提示
- 函数必须具有标准 Google 风格的 docstring（换句话说，是一个初始函数描述，后跟一个 `Args:` 块，描述参数，除非函数没有参数）
- 不要在 `Args:` 块中包含类型。换句话说，写 `a: The first number to multiply`，而不是 `a (int): The first number to multiply`。类型提示应放在函数头中
- 函数可以具有返回类型和 `Returns:` 块在 docstring 中。然而，这些是可选的，因为大多数工具使用模型会忽略它们

### 将工具结果传递给模型

上面的示例代码足以列出模型的可用工具，但如果模型实际想要使用其中一个工具，你应该：

1. 解析模型的输出以获取工具名称和参数
2. 将模型的工具调用添加到对话中
3. 使用这些参数调用相应的函数
4. 将结果添加到对话中

### 完整的工具使用示例

在这个示例中，我们将使用 `Hermes-2-Pro` 模型，因为它是撰写本文时在其尺寸类别中性能最高的工具使用模型之一。如果你有内存，可以考虑使用更大的模型，如 [Command-R](<url id="" type="url" status="" title="" wc="">https://huggingface.co/CohereForAI/c4ai-command-r-v01</url> ) 或 [Mixtral-8x22B](<url id="" type="url" status="" title="" wc="">https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1</url> )，它们都支持工具使用并提供更强的性能。

首先，加载模型和分词器：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
)
```

接下来，定义工具列表：

```python
def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]
```

现在，为机器人设置一个对话：

```python
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

现在，应用聊天模板并生成响应：

```python
inputs = tokenizer.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

我们得到以下结果：

```python
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

模型已使用有效参数调用了函数，格式符合函数 docstring 的要求。它推断我们很可能指的是法国的巴黎，并且记得作为国际单位制的发源地，法国的温度应该以摄氏度显示。

上面的输出格式特定于我们在此示例中使用的 `Hermes-2-Pro` 模型。其他模型可能会发出不同的工具调用格式，你可能需要在此步骤进行一些手动解析。例如，`Llama-3.1` 模型将发出稍有不同的 JSON，其中包含 `parameters` 而不是 `arguments`。无论模型输出的格式如何，你都应该按照以下格式将工具调用添加到对话中，包含 `tool_calls`、`function` 和 `arguments` 键。

接下来，将模型的工具调用添加到对话中：

```python
tool_call = {
    "name": "get_current_temperature",
    "arguments": {"location": "Paris, France", "unit": "celsius"}
}
messages.append(
    {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": tool_call}]
    }
)
```

如果你熟悉 OpenAI API，请注意一个重要区别——`tool_call` 是一个字典，但在 OpenAI API 中它是一个 JSON 字符串。传递字符串可能会导致错误或奇怪的模型行为！

现在我们已将工具调用添加到对话中，可以调用函数并将结果添加到对话中。由于我们在此示例中仅使用了一个始终返回 22.0 的占位函数，可以直接添加该结果。

```python
messages.append(
    {
        "role": "tool",
        "name": "get_current_temperature",
        "content": "22.0"
    }
)
```

一些模型架构（特别是 Mistral/Mixtral）还需要一个 `tool_call_id`，这应该是一个随机生成的 9 个字母数字字符，并分配给工具调用字典的 `id` 键。相同的键还应分配给下面工具响应字典的 `tool_call_id` 键，以便将工具调用与工具响应匹配。因此，对于 Mistral/Mixtral 模型，上面的代码将是：

```python
tool_call_id = "9Ae3bDc2F"  # 随机 ID，9 个字母数字字符
tool_call = {
    "name": "get_current_temperature",
    "arguments": {"location": "Paris, France", "unit": "celsius"}
}
messages.append(
    {
        "role": "assistant",
        "tool_calls": [{"type": "function", "id": tool_call_id, "function": tool_call}]
    }
)
```

以及

```python
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": "get_current_temperature",
        "content": "22.0"
    }
)
```

最后，让助手读取函数输出并继续与用户聊天：

```python
inputs = tokenizer.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

我们得到以下结果：

```python
The current temperature in Paris, France is 22.0 ° Celsius.<|im_end|>
```

尽管这是一个使用占位工具和单次调用的简单演示，但相同的技术适用于多个真实工具和更长的对话。这可以是通过实时信息、计算工具（如计算器）或大型数据库访问来扩展对话代理功能的强大力量。

### 理解工具模式

你传递给 `apply_chat_template` 的 `tools` 参数的每个函数都会被转换为一个 [JSON 模式](<url id="" type="url" status="" title="" wc="">https://json-schema.org/learn/getting-started-step-by-step</url> )。这些模式随后会被传递给模型聊天模板。换句话说，工具使用模型不会直接看到你的函数，也永远不会看到它们的实际代码。它们关心的是函数的**定义**和需要传递给它们的**参数**——它们关心工具的作用以及如何使用它们，而不是它们的工作原理！由你来读取它们的输出，检测它们是否请求使用工具，将参数传递给工具函数，并以聊天的形式返回响应。

如果遵循上述规范，生成 JSON 模式以传递给模板应该是自动且不可见的，但如果你遇到问题，或者只是想对转换过程有更多的控制，你可以手动处理转换。以下是一个手动模式转换的示例。

```python
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

这将生成以下内容：

```json
{
  "type": "function",
  "function": {
    "name": "multiply",
    "description": "A function that multiplies two numbers",
    "parameters": {
      "type": "object",
      "properties": {
        "a": {
          "type": "number",
          "description": "The first number to multiply"
        },
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      },
      "required": ["a", "b"]
    }
  }
}
```

如果愿意，你可以编辑这些模式，甚至完全不使用 `get_json_schema` 自己编写它们。可以直接将 JSON 模式传递给 `apply_chat_template` 的 `tools` 参数——这为你定义更复杂函数的精确模式提供了很大的灵活性。但请注意——你的模式越复杂，模型在处理它们时越容易混淆！我们建议尽可能使用简单的函数签名，尽量减少参数（尤其是复杂、嵌套的参数）。

以下是一个手动定义模式并直接传递给 `apply_chat_template` 的示例：

```python
# 一个不需要参数的简单函数
current_time = {
  "type": "function",
  "function": {
    "name": "current_time",
    "description": "Get the current local time as a string.",
    "parameters": {
      'type': 'object',
      'properties': {}
    }
  }
}

# 一个需要两个数值参数的更完整函数
multiply = {
  'type': 'function',
  'function': {
    'name': 'multiply',
    'description': 'A function that multiplies two numbers',
    'parameters': {
      'type': 'object',
      'properties': {
        'a': {
          'type': 'number',
          'description': 'The first number to multiply'
        },
        'b': {
          'type': 'number', 'description': 'The second number to multiply'
        }
      },
      'required': ['a', 'b']
    }
  }
}

model_input = tokenizer.apply_chat_template(
    messages,
    tools = [current_time, multiply]
)
```

## 高级：检索增强生成

“检索增强生成”或“RAG” LLMs 可以在回应查询之前搜索文档语料库中的信息。这允许模型大幅扩展其知识库，超越其有限的上下文大小。我们对 RAG 模型的建议是，它们的模板应接受一个 `documents` 参数。这应该是一个文档列表，每个“文档”是一个包含 `title` 和 `contents` 键的单个字典，两者都是字符串。由于这种格式比工具使用的 JSON 模式简单得多，因此不需要辅助函数。

以下是一个 RAG 模板的示例：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_id = "CohereForAI/c4ai-command-r-v01-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
device = model.device # 获取模型加载的设备

# 定义对话输入
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]

# 定义检索生成的文档
documents = [
    {
        "title": "The Moon: Our Age-Old Foe",
        "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
    },
    {
        "title": "The Sun: Our Age-Old Friend",
        "text": "Although often underappreciated, the sun provides several notable benefits..."
    }
]

# 使用 RAG 模板对对话和文档进行分词，返回 PyTorch 张量
input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

# 生成响应
gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

# 解码并打印生成的文本以及生成提示
gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

RAG 的 `documents` 输入尚未被广泛支持，许多模型的聊天模板会直接忽略此输入。

要验证模型是否支持 `documents` 输入，可以阅读其 `Model Card`，或者 `print(tokenizer.chat_template)` 查看 `documents` 键是否在任何地方被使用。

不过，Cohere 的 [Command-R](<url id="" type="url" status="" title="" wc="">https://huggingface.co/CohereForAI/c4ai-command-r-08-2024</url> ) 和 [Command-R+](<url id="" type="url" status="" title="" wc="">https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024</url> ) 模型通过其 `rag` 聊天模板支持此功能。你可以在它们的模型卡中查看使用此功能的更多示例。

## 高级：聊天模板是如何工作的？

模型的聊天模板存储在 `tokenizer.chat_template` 属性中。如果没有设置聊天模板，则使用该模型类的默认模板。让我们看看一个简化的 `Zephyr` 聊天模板：

```jinja2
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

如果你以前从未见过这样的内容，这是一个 [Jinja 模板](<url id="" type="url" status="" title="" wc="">https://jinja.palletsprojects.com/en/3.1.x/templates/</url> )。Jinja 是一种模板语言，允许你编写生成文本的简单代码。在很多方面，代码和语法类似于 Python。在纯 Python 中，这个模板看起来像这样：

```python
for message in messages:
    print(f'<|{message["role"]}|>')
    print(message['content'] + eos_token)
if add_generation_prompt:
    print('<|assistant|>')
```

实际上，模板做了三件事：

1. 对于每条消息，打印角色，用 `<|` 和 `|>` 包裹，例如 `<|user|>` 或 `<|assistant|>`。
2. 接下来，打印消息内容，后跟序列结束标记。
3. 最后，如果设置了 `add_generation_prompt`，打印助手标记，以便模型知道开始生成助手响应。

这是一个相当简单的模板，但 Jinja 为你提供了做更复杂事情的灵活性！让我们看看一个可以类似于 LLaMA 格式化输入的 Jinja 模板（注意，实际的 LLaMA 模板包括对默认系统消息的处理以及稍微不同的系统消息处理方式——不要在实际代码中使用这个模板！）：

```jinja2
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + message['content'] + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
```

希望如果你仔细看看这个模板，你能理解它在做什么——它根据每条消息的角色添加特定的标记，如 `[INST]` 和 `[/INST]`。用户、助手和系统消息因为包裹它们的标记而对模型清晰可辨。

## 高级：添加和编辑聊天模板

### 如何创建聊天模板？

很简单，只需编写一个 Jinja 模板并设置 `tokenizer.chat_template`。你可能会发现从另一个模型的现有模板开始并根据需要编辑它更容易！例如，我们可以取上面的 LLaMA 模板并添加 “[ASST]” 和 “[/ASST]” 到助手消息：

```jinja2
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
```

现在，只需设置 `tokenizer.chat_template` 属性。下次你使用 [apply_chat_template()](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template</url> ) 时，它将使用你的新模板！此属性将保存在 `tokenizer_config.json` 文件中，因此你可以使用 [push_to_hub()](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub</url> ) 将你的新模板上传到 Hub，并确保每个人都为你的模型使用正确的模板！

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # 修改系统标记
tokenizer.chat_template = template  # 设置新模板
tokenizer.push_to_hub("model_name")  # 将新模板上传到 Hub！
```

使用你的聊天模板的方法 [apply_chat_template()](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template</url> ) 被 [TextGenerationPipeline](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline</url> ) 类调用，因此一旦你设置了正确的聊天模板，你的模型将自动与 [TextGenerationPipeline](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline</url> ) 兼容。

如果你正在为聊天微调模型，除了设置聊天模板外，你还应该将任何新的聊天控制标记添加为分词器中的特殊标记。特殊标记永远不会被拆分，确保你的控制标记始终被处理为单个标记，而不是被分词为多个部分。你还应将分词器的 `eos_token` 属性设置为模板中标记助手生成结束的标记。这将确保文本生成工具能够正确判断何时停止生成文本。

### 为什么有些模型有多个模板？

有些模型在不同用例中使用不同的模板。例如，它们可能为普通聊天使用一个模板，为工具使用或检索增强生成使用另一个模板。在这种情况下，`tokenizer.chat_template` 是一个字典。这可能会引起一些混淆，因此我们建议尽可能使用单个模板。你可以使用 Jinja 语句（如 `if tools is defined`）和 `{% macro %}` 定义来轻松将多个代码路径封装到一个模板中。

当分词器有多个模板时，`tokenizer.chat_template` 将是一个 `dict`，其中每个键是模板的名称。`apply_chat_template` 方法对某些模板名称有特殊处理：通常情况下，它会查找名为 `default` 的模板，如果找不到则会报错。然而，如果用户传递了 `tools` 参数且存在名为 `tool_use` 的模板，它将使用该模板。要访问其他名称的模板，请将模板名称传递给 `apply_chat_template()` 的 `chat_template` 参数。

我们发现这可能会让用户感到困惑——因此，如果你正在编写模板，我们建议尽可能将所有内容放在一个模板中！

### 我应该使用什么模板？

当为已经训练用于聊天的模型设置模板时，你应该确保模板完全匹配模型在训练期间看到的消息格式，否则可能会导致性能下降。即使你进一步训练模型，保持聊天标记不变通常会获得最佳性能。这与分词非常相似——在推理或微调时，精确匹配训练期间使用的分词通常会获得最佳性能。

如果你从头开始训练模型，或者为聊天微调基础语言模型，你有很大的自由度来选择合适的模板！LLMs 足够智能，可以学会处理许多不同的输入格式。一个流行的选择是 `ChatML` 格式，这对于许多用例来说是一个灵活的选择。它看起来像这样：

```jinja2
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

如果你喜欢这个模板，这里是一行代码形式，可以直接复制到你的代码中。这一行代码还方便地支持了 [生成提示](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/chat_templating#what-are-generation-prompts</url> )，但请注意，它不会添加 BOS 或 EOS 标记！如果你的模型期望这些标记，它们不会由 `apply_chat_template` 自动添加——换句话说，文本将以 `add_special_tokens=False` 进行分词。这是为了避免模板和 `add_special_tokens` 逻辑之间的潜在冲突。如果你的模型期望特殊标记，请确保将它们添加到模板中！

```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

此模板将每条消息用 `<|im_start|>` 和 `<|im_end|>` 标记包裹，并简单地将角色写为字符串，这允许你在训练中使用灵活的角色。输出看起来像这样：

```python
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

“user”、“system” 和 “assistant” 角色是聊天的标准角色，我们建议在适用时使用它们，特别是如果你希望你的模型与 [TextGenerationPipeline](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline</url> ) 兼容。然而，你并不限于这些角色——模板非常灵活，任何字符串都可以作为角色。

### 我想添加一些聊天模板！我该如何开始？

如果你有任何聊天模型，你应该设置它们的 `tokenizer.chat_template` 属性，并使用 [apply_chat_template()](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template</url> ) 进行测试，然后将更新后的分词器推送到 Hub。即使你不是模型所有者，这也适用——如果你正在使用一个聊天模板为空或仍然使用默认类模板的模型，请向模型仓库提交一个 [pull request](<url id="" type="url" status="" title="" wc="">https://huggingface.co/docs/hub/repositories-pull-requests-discussions</url> )，以便正确设置此属性！

一旦设置了该属性，就完成了！`tokenizer.apply_chat_template` 现在可以正确用于该模型，这意味着它也自动在 `TextGenerationPipeline` 等地方受支持！

通过确保模型具有此属性，我们可以确保整个社区都能充分利用开源模型的全部功能。格式不匹配的问题已经困扰该领域太久，并默默地损害了性能——是时候结束这一切了！

## 高级：模板编写技巧

开始编写 Jinja 模板的最简单方法是查看一些现有的模板。你可以使用 `print(tokenizer.chat_template)` 查看任何聊天模型正在使用的模板。通常，支持工具使用的模型比其他模型具有更复杂的模板——因此，当你刚开始时，它们可能不是学习的好例子！你还可以查看 [Jinja 文档](<url id="" type="url" status="" title="" wc="">https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis</url> ) 以了解 Jinja 的一般格式和语法。

`transformers` 中的 Jinja 模板与其他地方的 Jinja 模板相同。需要了解的主要内容是，对话历史将在模板中作为一个名为 `messages` 的变量可用。你将能够像在 Python 中一样访问 `messages`，这意味着你可以使用 `{% for message in messages %}` 遍历它，或者使用 `{{ messages[0] }}` 访问单个消息。

你还可以使用以下技巧编写清晰、高效的 Jinja 模板：

### 修剪空白

默认情况下，Jinja 会打印块前后任何空白。这可能对聊天模板造成问题，因为聊天模板通常希望对空白非常精确！为了避免这种情况，我们强烈建议像这样编写模板：

```jinja2
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

而不是像这样：

```python
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

添加 `-` 将剥离块前后的任何空白。第二个示例看起来无害，但换行和缩进可能会包含在输出中，这可能不是你想要的！

### 特殊变量

在模板中，你将能够访问几个特殊变量。其中最重要的是 `messages`，它包含对话历史作为消息字典的列表。然而，还有几个其他变量。并非每个变量都会在每个模板中使用。最常见的其他变量是：

- `tools` 包含一个 JSON 模式格式的工具列表。如果没有传递工具，它将为 `None` 或未定义。
- `documents` 包含一个用于检索增强生成的文档列表，格式为 `{"title": "Title", "contents": "Contents"}`。如果没有传递文档，它将为 `None` 或未定义。
- `add_generation_prompt` 是一个布尔值，如果用户请求生成提示，则为 `True`，否则为 `False`。如果设置了此值，你的模板应在对话末尾添加助手消息的标题。如果你的模型没有特定的助手消息标题，你可以忽略此标志。
- **特殊标记**，如 `bos_token` 和 `eos_token`。这些从 `tokenizer.special_tokens_map` 中提取。每个模板中可用的确切标记会根据父分词器而有所不同。

实际上，你可以将任何 `kwarg` 传递给 `apply_chat_template`，它将在模板中作为变量可用。通常，我们建议尽量使用上述核心变量，因为这会使你的模型更难使用，如果用户必须编写自定义代码来传递模型特定的 `kwargs`。然而，我们意识到这个领域发展迅速，所以如果你有不符合核心 API 的新用例，可以自由使用新的 `kwarg`！如果一个新的 `kwarg` 变得常见，我们可能会将其提升到核心 API 中，并创建一个标准的、有文档记录的格式。

### 可调用函数

在模板中，你还可以使用几个可调用函数。这些函数是：

- `raise_exception(msg)`：引发 `TemplateException`。这在调试时很有用，也可以告诉用户他们的操作不受模板支持。
- `strftime_now(format_str)`：等同于 Python 中的 `datetime.now().strftime(format_str)`。这用于以特定格式获取当前日期/时间，有时会包含在系统消息中。

### 与非 Python Jinja 的兼容性

有多种语言的 Jinja 实现。它们通常具有相同的语法，但一个关键区别是，当你在 Python 中编写模板时，可以使用 Python 方法，例如对字符串使用 `.lower()` 或对字典使用 `.items()`。这在有人尝试在非 Python 实现的 Jinja 上使用你的模板时会中断。非 Python 实现尤其在部署环境中很常见，JS 和 Rust 在这些环境中非常流行。

不要惊慌！你可以对模板进行一些简单的更改，以确保它们在所有 Jinja 实现中兼容：

- 用 Jinja 过滤器替换 Python 方法。这些通常具有相同的名称，例如 `string.lower()` 变为 `string|lower`，`dict.items()` 变为 `dict|items`。一个显著的变化是 `string.strip()` 变为 `string|trim`。有关 Jinja 文档中内置过滤器的列表，请参阅 [内置过滤器](<url id="" type="url" status="" title="" wc="">https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters</url> )。
- 用 Python 特定的 `True`、`False` 和 `None` 替换为 `true`、`false` 和 `none`。
- 直接渲染字典或列表在其他实现中可能会产生不同的结果（例如，字符串条目可能从单引号变为双引号）。添加 `tojson` 过滤器可以帮助确保一致性。

### 编写生成提示

我们之前提到过，`add_generation_prompt` 是一个特殊变量，将在模板中可用，并由用户设置的 `add_generation_prompt` 标志控制。如果你的模型期望助手消息的标题，那么你的模板必须在 `add_generation_prompt` 设置时支持添加标题。

以下是一个以 ChatML 格式的消息模板示例，支持生成提示：

```jinja2
{{- bos_token }}
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

助手标题的确切内容取决于你的具体模型，但它始终应该是**表示助手消息开始的字符串**，以便用户应用你的模板并生成文本时，模型将写入助手响应。请注意，一些模型不需要生成提示，因为助手消息总是在用户消息后立即开始。这在 LLaMA 和 Mistral 模型中尤为常见，助手消息在结束用户消息的 `[/INST]` 标记后立即开始。在这种情况下，模板可以忽略 `add_generation_prompt` 标志。

生成提示很重要！如果模型需要生成提示但模板中未设置，模型生成的内容可能会严重退化，或者模型可能会表现出继续最后一条用户消息等异常行为！

### 编写和调试较大的模板

当此功能推出时，大多数模板都很小，相当于 Jinja 的“一行代码”脚本。然而，随着新模型和工具使用、RAG 等功能的出现，一些模板可能长达 100 行或更多。在编写这样的模板时，最好在单独的文件中使用文本编辑器编写。你可以轻松地将聊天模板提取到文件中：

```python
open("template.jinja", "w").write(tokenizer.chat_template)
```

或者将编辑后的模板加载回分词器：

```python
tokenizer.chat_template = open("template.jinja").read()
```

额外的好处是，当你在单独的文件中编写长的多行模板时，该文件中的行号将与模板解析或执行错误中的行号完全对应。这将使识别问题来源变得容易得多。

### 为工具编写模板

尽管聊天模板不强制特定的工具 API（或任何东西），但我们建议模板作者尽可能遵循标准 API。聊天模板的全部目的是允许代码在模型之间移植，因此偏离标准工具 API 意味着用户必须编写自定义代码才能使用工具与你的模型。有时这是不可避免的，但通常通过巧妙的模板设计，可以使标准 API 正常工作！

下面，我们将列出标准 API 的元素，并提供编写与之兼容的模板的提示。

#### 工具定义

你的模板应期望变量 `tools` 要么为 null（如果没有传递工具），要么是一个 JSON 模式字典的列表。我们的聊天模板方法允许用户将工具作为 JSON 模式或 Python 函数传递，但当传递函数时，我们会自动生成 JSON 模式并将其传递给你的模板。因此，你的模板接收到的 `tools` 变量始终是一个 JSON 模式列表。以下是一个工具 JSON 模式的示例：

```json
{
  "type": "function",
  "function": {
    "name": "multiply",
    "description": "A function that multiplies two numbers",
    "parameters": {
      "type": "object",
      "properties": {
        "a": {
          "type": "number",
          "description": "The first number to multiply"
        },
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      },
      "required": ["a", "b"]
    }
  }
}
```

以下是一个在聊天模板中处理工具的示例代码。请注意，这只是针对特定格式的示例——你的模型可能需要不同的格式！

```jinja2
{%- if tools %}
    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}
{%- endif %}
```

你的模板渲染的工具调用的具体标记和描述应与模型训练时使用的格式匹配。没有要求你的**模型**理解 JSON 模式输入，只需要你的模板可以将 JSON 模式转换为模型的格式。例如，[Command-R](<url id="" type="url" status="" title="" wc="">https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024</url> ) 是使用 Python 函数头定义工具训练的，但 Command-R 工具模板接受 JSON 模式，内部转换类型，并将输入工具渲染为 Python 函数头。你可以用模板做很多事情！

#### 工具调用

如果存在工具调用，它们将作为具有“assistant”角色的消息的列表附加。请注意，`tool_calls` 始终是一个列表，即使大多数工具调用模型一次只支持单个工具调用，这意味着列表通常只有一个元素。以下是一个包含工具调用的消息字典示例：

```python
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": {
          "a": 5,
          "b": 6
        }
      }
    }
  ]
}
```

一个常见的处理模式可能是这样的：

```python
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
        {%- endif %}
    {%- endfor %}
{%- endif %}
```

同样，你应该使用模型期望的格式和特殊标记来渲染工具调用。

#### 工具响应

工具响应的格式很简单：它们是一个具有“tool”角色的消息字典，一个提供调用函数名称的“name”键，以及包含工具调用结果的“content”键。以下是一个工具响应的示例：

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

你不需要使用工具响应中的所有键。例如，如果模型不期望工具响应中包含函数名称，那么渲染它可以像这样一样简单：

```python
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

同样，请记住，实际的格式和特殊标记是模型特定的——你应该非常小心地确保标记、空白以及所有内容完全匹配模型训练时使用的格式！
