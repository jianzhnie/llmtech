# 如何用 Langchain 和 OpenAI-api 复制 ChatGPT？

众所周知，ChatGPT 目前能够取得令人印象深刻的壮举。很可能许多人都有在他们自己的项目中使用该技术的想法。

## 聊天GPT

[ChatGPT](https://openai.com/blog/chatgpt/) 在其设计中使用了 GPT-3 模型，并在此基础上开发了一个新模型。因此，新模型的输出结果往往与 GPT-3 相似。在撰写本文时，该`text-davinci-002-render`模型已在新模型的 ChatGPT 中使用，但目前尚未向公众开放。

虽然 ChatGPT 可能不是开创性的，但它提供了一个利用现有技术的新界面。通过利用强大的提示和高效的记忆窗口。因此，我们可以使用 LLM Chain 方法复制其功能，而不是破解 ChatGPT 的非官方 API。

## Langchain

Langchain 是一个新的 python 包，它为链提供了一个标准接口，与其他工具的大量集成，以及用于常见应用程序的端到端链。

LangChain 旨在协助四个主要领域，此处按复杂性递增的顺序列出：

1. LLM和Prompt
2. 链条
3. 代理商
4. 记忆

在[官方文档](https://langchain.readthedocs.io/en/latest/?)中了解有关 langchain 的更多信息。

## 安装

要使用 langchain 包，您可以从 pypi 安装它。

```bash
pip install langchain
```

要从 langchain 获取最新更新，您可以使用这种安装方法。

```bash
pip install "git+https://github.com/hwchase17/langchain.git"
```

更多安装选项请阅读[此处](https://langchain.readthedocs.io/en/latest/installation.html)。

## 示例项目

您可以使用 ChatGPT 做很多事情，其中一个有趣的事情是为学生作业建立问答。所以这次我们将创建 AI 版的[Brainly](https://brainly.com/)。

这就是我们将从 ChatGPT 中得到的。

![chatgpt-result-example](https://ahmadrosid.com/images/chatgpt-result-example.png)

这是 langchain 的提示。

```py
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

llm = OpenAI(temperature=.7)
template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
Question: {text}
Answer:
"""
prompt_template = PromptTemplate(input_variables=["text"], template=template)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
answer = answer_chain.run("What is the formula for Gravitational Potential Energy (GPE)?")
print(answer)
```

这是我们使用 langchain 从 GPT-3 获得的结果。

```shell
The formula for Gravitational Potential Energy (GPE) is GPE = mgh, where m is the mass of an object, g is the acceleration due to gravity, and h is the height of the object. For example, if an object with a mass of 10 kg is at a height of 5 meters, then the GPE would be GPE = 10 x 9.8 x 5 = 490 Joules.
```

## 聊天机器人

如果你需要创建像 AI 这样的聊天机器人，你可以使用 langchain 的内存。这是如何操作的示例。

```py
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
{chat_history}
Human: {question}
AI:
"""
prompt_template = PromptTemplate(input_variables=["chat_history","question"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt_template,
    verbose=True,
    memory=memory,
)

llm_chain.predict(question="What is the formula for Gravitational Potential Energy (GPE)?")

result = llm_chain.predict(question="What is Joules?")
print(result)
```

结果会是这样的。

```bash
$ python3 memory.py

> Entering new LLMChain chain...
Prompt after formatting:
You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.

Human: What is the formula for Gravitational Potential Energy (GPE)?
AI:

> Finished LLMChain chain.

> Entering new LLMChain chain...
Prompt after formatting:
You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.

Human: What is the formula for Gravitational Potential Energy (GPE)?
AI:
The formula for Gravitational Potential Energy (GPE) is GPE = mgh, where m is the mass of the object, g is the acceleration due to gravity, and h is the height of the object.

For example, if an object has a mass of 10 kg and is at a height of 5 meters, then the gravitational potential energy of the object is GPE = 10 kg x 9.8 m/s2 x 5 m = 490 Joules.
Human: What is Joules?
AI:

> Finished LLMChain chain.
Joules (J) is the SI unit of energy. It is defined as the amount of energy required to move an object of one kilogram at a speed of one meter per second. It is also equal to the work done when a force of one Newton is applied to an object and moved one meter in the direction of the force.
```

## 结论

ChatGPT 是一个基于 GPT-3 的聊天机器人，目前没有官方 API。使用 LangChain，开发人员可以复制 ChatGPT 的功能，例如创建聊天机器人或问答系统，而无需使用非官方 API。

LangChain 为常见应用程序提供标准接口、大量集成和端到端链。它可以从 pypi 安装，更多信息可以在官方文档中找到。
