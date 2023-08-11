# List of Open Sourced Fine-Tuned Large Language Models (LLM)

> An incomplete list of open-sourced fine-tuned Large Language Models (LLM) you can run locally on your computer

This is an incomplete list of open-sourced fine-tuned Large Language Models (LLMs) that runs on your local computer, and my attempt to maintain a list since as many as three models are announced on a daily basis.

*I haven’t listed them all because you can literally create these models for less than $100. Cabrita, which is one of the models listed here was created for $8 — I find it hard to believe. I am still thinking about whether or not I should create BritneyGPT, but I did create the training dataset for about $20, and it would cost me an additional $50 to use GPU services. I have even thought about the name for the article — “It’s BritneyGPT, B******!”*

According to the documentation, you can run these models on a PC with different levels of hardware. For most people, your best bet is **llama.cpp** since it supports seven models and runs on moderately specced PCs:

*LLaMA | Alpaca | GPT4All | Chinese LLaMA/Alpaca | Vigogne (French) | Vicuna | Koala | OpenBuddy (Multilingual)*

The list is a work in progress where I tried to group them by the Foundation Models where they are:

*BigCode’s StarCoder | BigScience’s BLOOM | Cerebras’ Cerebras-GPT | EleutherAI’s GPT-J, GPT-NeoX, Polyglot, and Pythia | GLM | Google’s Flamingo, FLAN, and PaLM | H2O.ai’s h2ogpt | Meta’s GALACTICA, LLaMA, and XGLM | Mosaic ML’s MPT | Nvidia’s NeMo | OpenLLaMA | Replit’s Code | RWKV | StabilityAI’s StableLM | Together’s RedPajama-INCITE*

They are subgrouped by the list of projects that are reproductions of or based on those Foundation Models.

**Updates:**

- 03/2023: Added HuggingGPT | Vicuna/FastChat
- 04/2023: Added “A Survey of Large Language Models” | “LLMMaps — A Visual Metaphor for Stratified Evaluation of Large Language Models” | Baize | Koala | Segment Anything | Galpaca | GPT-J-6B instruction-tuned on Alpaca-GPT4 | GPTQ-for-LLaMA | List of all Foundation Models | Dolly 2.0 | StackLLaMA | GPT4All-J | Palmyra Base 5B | Camel 🐪 5B | StableLM | h2oGPT | The Bloke alpaca-lora-65B-GGML | OpenAssistant Models | StableVicuna | FastChat-T5 | couchpotato888 | GPT4-x-Alpaca | LLaMA Adapter V2 | WizardLM, | A brief history of LLaMA models (Resources section)
- 05/2023: Added OpenLLaMA | BigCode StarCoder (Hugging Face + ServiceNow) | Replit-Code (Replit) | Pygmalion-7b | AlpacaGPT4-LoRA-7B-OpenLLaMA | Nvidia GPT-2B-001 | The Bloke’s StableVicuna-13B-GPTQ | OpenAlpaca | crumb’s Hugging Face website | Teknium’s Hugging Face website | Knut Jägersberg’s Hugging Face website | SemiAnalysis article by Luke Sernau (a senior software engineer at Google) | Mosaic ML’s MPT-7B | gpt4-x-vicuna-13b | LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions | Vigogne | Chinese-LLaMA-Alpaca | OpenBuddy — Open Multilingual Chatbot for Everyone | Chatbot Arena | Together’s RedPajama-INCITE 3B and 7B | Ahead of AI #8: The Latest Open Source LLMs and Datasets (Resources section) | PaLM (Concept of Mind) | digitous Hugging Face website | Hugging Face’s Open LLM Leaderboard | A’eala’s Hugging Face website | chavinlo’s Hugging Face website | eachadea’s Hugging Face website | chainyo’s Hugging Face website | KoboldAI’s Hugging Face website | Baize V2

# LLaMA (Meta)

> Stanford Alpaca: An Instruction-following LLaMA Model.

- LLaMA Website: [Introducing LLaMA: A foundational, 65-billion-parameter language model (facebook.com)](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
- Alpaca Website: https://crfm.stanford.edu/2023/03/13/alpaca.html
- Alpaca GitHub: https://github.com/tatsu-lab/stanford_alpaca
- Commercial Use: No

Here is a list of reproductions of or based on Meta’s LLaMA or Stanford Alpaca project:

*Alpaca.cpp | Alpaca-LoRA | AlpacaGPT4-LoRA-7B-OpenLLaMA | Baize | Cabrita | Chinese-LLaMA-Alpaca | Chinese-Vicuna | GPT4-x-Alpaca | gpt4-x-vicuna-13b | GPT4All | GPTQ-for-LLaMA | Koala | llama.cpp | LLaMA-Adapter V2 | Lit-LLaMA | OpenAlpaca | OpenBuddy — Open Multilingual Chatbot for Everyone | Pygmalion-7b | StackLLaMA | StableVicuna | The Bloke alpaca-lora-65B-GGML/StableVicuna-13B-GPTQ/WizardLM-7B-uncensored-GPTQ | Vicuna | Vigogne | WizardLM*

## Alpaca.cpp

> Run a fast ChatGPT-like model locally on your device. The screencast below is not sped up and running on an M2 Macbook Air with 4GB of weights.

- GitHub: [antimatter15/alpaca.cpp: Locally run an Instruction-Tuned Chat-Style LLM (github.com)](https://github.com/antimatter15/alpaca.cpp)

## Alpaca-LoRA

> This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf). We provide an Instruct model of similar quality to `text-davinci-003` that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research), and the code is easily extended to the `13b`, `30b`, and `65b` models.

- GitHub: [tloen/alpaca-lora: Instruct-tune LLaMA on consumer hardware (github.com)](https://github.com/tloen/alpaca-lora)
- Demo: [Alpaca-LoRA — a Hugging Face Space by tloen](https://huggingface.co/spaces/tloen/alpaca-lora)

## AlpacaGPT4-LoRA-7B-OpenLLaMA

- Hugging Face: https://huggingface.co/LLMs
- LLMs Models: https://huggingface.co/LLMs

## Baize V2

> Baize V2 is an open-source chat model fine-tuned with LoRA. It uses 100k dialogs generated by letting ChatGPT chat with itself. We also use Alpaca’s data to improve its performance. We have released 7B, and 13B models.

- GitHub: [project-baize/baize: Baize is an open-source chatbot trained with ChatGPT self-chatting data, developed by researchers at UCSD and Sun Yat-sen University. (github.com)](https://github.com/project-baize/baize)
- Paper: [2304.01196.pdf (arxiv.org)](https://arxiv.org/pdf/2304.01196.pdf)

## Cabrita

> A portuguese finetuned instruction LLaMA

- GitHub: https://github.com/22-hours/cabrita

## Chinese-LLaMA-Alpaca

> In order to promote the open research of large models in the Chinese NLP community, this project open sourced the Chinese LLaMA model and the Alpaca large model with fine-tuned instructions. Based on the original LLaMA, these models expand the Chinese vocabulary and use Chinese data for secondary pre-training, which further improves the basic semantic understanding of Chinese. At the same time, the Chinese Alpaca model further uses Chinese instruction data for fine-tuning, which significantly improves the model’s ability to understand and execute instructions. For details, please refer to the technical report (Cui, Yang, and Yao, 2023).

- GitHub: https://github.com/ymcui/Chinese-LLaMA-Alpaca

## Chinese-Vicuna

> A Chinese Instruction-following LLaMA-based Model

- GitHub: [Facico/Chinese-Vicuna: Chinese-Vicuna: A Chinese Instruction-following LLaMA-based Model — — 一个中文低资源的llama+lora方案，结构参考alpaca (github.com)](https://github.com/Facico/Chinese-Vicuna)

## GPT4-x-Alpaca

> GPT4-x-Alpaca is a LLaMA 13B model fine-tuned with a collection of GPT4 conversations, GPTeacher. There’s not a lot of information on its training and performance.

- Hugging Face: [chavinlo/gpt4-x-alpaca · Hugging Face](https://huggingface.co/chavinlo/gpt4-x-alpaca)

## gpt4-x-vicuna-13b

> As a base model used https://huggingface.co/eachadea/vicuna-13b-1.1. Finetuned on Teknium’s GPTeacher dataset, unreleased Roleplay v2 dataset, GPT-4-LLM dataset, and Nous Research Instruct Dataset. Approx 180k instructions, all from GPT-4, all cleaned of any OpenAI censorship/”As an AI Language Model” etc.

- Hugging Face: [NousResearch/gpt4-x-vicuna-13b · Hugging Face](https://huggingface.co/NousResearch/gpt4-x-vicuna-13b)

## GPT4All

> Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa.

- GitHub: [nomic-ai/gpt4all: gpt4all: a chatbot trained on a massive collection of clean assistant data including code, stories and dialogue (github.com)](https://github.com/nomic-ai/gpt4all)
- GitHub: [nomic-ai/pyllamacpp: Official supported Python bindings for llama.cpp + gpt4all (github.com)](https://github.com/nomic-ai/pyllamacpp)
- Review: [Is GPT4All your new personal ChatGPT? — YouTube](https://www.youtube.com/watch?v=GhRNIuTA2Z0)

## GPTQ-for-LLaMA

> 4 bits quantization of [LLaMA](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323). GPTQ is SOTA one-shot weight quantization method.

- GitHub: [qwopqwop200/GPTQ-for-LLaMa: 4 bits quantization of LLaMA using GPTQ (github.com)](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

## Koala

> Koala is a language model fine-tuned on top of LLaMA. [Check out the blogpost!](https://bair.berkeley.edu/blog/2023/04/03/koala/) This documentation will describe the process of downloading, recovering the Koala model weights, and running the Koala chatbot locally.

- Blog: [Koala: A Dialogue Model for Academic Research — The Berkeley Artificial Intelligence Research Blog](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- GitHub: [EasyLM/koala.md at main · young-geng/EasyLM (github.com)](https://github.com/young-geng/EasyLM/blob/main/docs/koala.md)
- Demo: [FastChat (lmsys.org)](https://chat.lmsys.org/?model=koala-13b)
- Review: [Investigating Koala a ChatGPT style Dialogue Model — YouTube](https://www.youtube.com/watch?v=A4rcKUZieEU)
- Review: [Running Koala for free in Colab. Your own personal ChatGPT? — YouTube](https://www.youtube.com/watch?v=kSLcedGSez8)

## llama.cpp

> Inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++

- GitHub: [ggerganov/llama.cpp: Port of Facebook’s LLaMA model in C/C++ (github.com)](https://github.com/ggerganov/llama.cpp)
- Supports three models: LLaMA, Alpaca, and GPT4All

## LLaMA-Adapter V2

> Official implementation of [‘LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention’](https://arxiv.org/pdf/2303.16199.pdf) and [‘LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model’](https://arxiv.org/pdf/2304.15010.pdf).

- GitHub: [ZrrSkywalker/LLaMA-Adapter: Fine-tuning LLaMA to follow Instructions within 1 Hour and 1.2M Parameters (github.com)](https://github.com/ZrrSkywalker/LLaMA-Adapter)

## Lit-LLaMA ️

> Independent implementation of [LLaMA](https://github.com/facebookresearch/llama) that is fully open source under the Apache 2.0 license. This implementation builds on [nanoGPT](https://github.com/karpathy/nanoGPT).

- GitHub: [Lightning-AI/lit-llama: Implementation of the LLaMA language model based on nanoGPT. Supports quantization, LoRA fine-tuning, pre-training. Apache 2.0-licensed. (github.com)](https://github.com/Lightning-AI/lit-llama)

## OpenAlpaca

> This is the repo for the OpenAlpaca project, which aims to build and share an instruction-following model based on OpenLLaMA. We note that, following OpenLLaMA, OpenAlpaca is permissively licensed under the Apache 2.0 license. This repo contains
>
> \- The data used for fine-tuning the model.
> \- The code for fine-tuning the model.
> \- The weights for the fine-tuned model.
> \- The example usage of OpenAlpaca.

- GitHub: [yxuansu/OpenAlpaca: OpenAlpaca: A Fully Open-Source Instruction-Following Model Based On OpenLLaMA (github.com)](https://github.com/yxuansu/OpenAlpaca)

## OpenBuddy — Open Multilingual Chatbot for Everyone

> OpenBuddy is a powerful open-source multilingual chatbot model aimed at global users, emphasizing conversational AI and seamless multilingual support for English, Chinese, and other languages. Built upon Facebook’s LLAMA model, OpenBuddy is fine-tuned to include an extended vocabulary, additional common characters, and enhanced token embeddings. By leveraging these improvements and multi-turn dialogue datasets, OpenBuddy offers a robust model capable of answering questions and performing translation tasks across various languages.

- GitHub: https://github.com/OpenBuddy/OpenBuddy

## Pygmalion-7b

> Pygmalion 7B is a dialogue model based on Meta’s LLaMA-7B. This is version 1. It has been fine-tuned using a subset of the data from Pygmalion-6B-v8-pt4, for those of you familiar with the project.

- Hugging Face: https://huggingface.co/PygmalionAI/pygmalion-7b

## StableVicuna

> We are proud to present [StableVicuna](https://huggingface.co/spaces/CarperAI/StableVicuna), the first large-scale open source chatbot trained via reinforced learning from human feedback (RHLF). StableVicuna is a further instruction fine tuned and RLHF trained version of Vicuna v0 13b, which is an instruction fine tuned [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) 13b model. For the interested reader, you can find more about [Vicuna here](https://vicuna.lmsys.org/).

- Website: [Stability AI releases StableVicuna, the AI World’s First Open Source RLHF LLM Chatbot — Stability AI](https://stability.ai/blog/stablevicuna-open-source-rlhf-chatbot)
- Hugging Face: [StableVicuna — a Hugging Face Space by CarperAI](https://huggingface.co/spaces/CarperAI/StableVicuna)
- Review: [StableVicuna: The New King of Open ChatGPTs? — YouTube](https://www.youtube.com/watch?v=m_xD0algP4k)

## StackLLaMA

> A [LlaMa model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai) trained on answers and questions on [Stack Exchange](https://stackexchange.com/) with RLHF through a combination of: Supervised Fine-tuning (SFT), Reward / preference modeling (RM), and Reinforcement Learning from Human Feedback (RLHF)

Website: https://huggingface.co/blog/stackllama

## The Bloke alpaca-lora-65B-GGML

> Quantised 4bit and 2bit GGMLs of [changsung’s alpaca-lora-65B](https://huggingface.co/chansung/alpaca-lora-65b) for CPU inference with [llama.cpp](https://github.com/ggerganov/llama.cpp).

- Hugging Face: [TheBloke/alpaca-lora-65B-GGML · Hugging Face](https://huggingface.co/TheBloke/alpaca-lora-65B-GGML)

## The Bloke’s StableVicuna-13B-GPTQ

> This repo contains 4bit GPTQ format quantised models of [CarterAI’s StableVicuna 13B](https://huggingface.co/CarperAI/stable-vicuna-13b-delta). It is the result of first merging the deltas from the above repository with the original Llama 13B weights, then quantising to 4bit using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

- Hugging Face: [TheBloke/stable-vicuna-13B-GPTQ · Hugging Face](https://huggingface.co/TheBloke/stable-vicuna-13B-GPTQ)

## The Bloke’s WizardLM-7B-uncensored-GPTQ

> These files are GPTQ 4bit model files for [Eric Hartford’s ‘uncensored’ version of WizardLM](https://huggingface.co/ehartford/WizardLM-7B-Uncensored). It is the result of quantising to 4bit using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa). Eric did a fresh 7B training using the WizardLM method, on [a dataset edited to remove all the “I’m sorry..” type ChatGPT responses](https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered).

- Hugging Face: [TheBloke/WizardLM-7B-uncensored-GPTQ · Hugging Face](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GPTQ)

## Vicuna (FastChat)

> An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality.

- GitHub: [lm-sys/FastChat: The release repo for “Vicuna: An Open Chatbot Impressing GPT-4” (github.com)](https://github.com/lm-sys/FastChat)
- Review: [Vicuna — 90% of ChatGPT quality by using a new dataset? — YouTube](https://www.youtube.com/watch?v=4VByC2NpV30)

## Vigogne

> This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) in French 🇫🇷 using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) provided by 🤗 Hugging Face’s [PEFT](https://github.com/huggingface/peft) library. In addition to the LoRA technique, we also use [LLM.int8()](https://arxiv.org/abs/2208.07339) provided by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to quantize pretrained language models (PLMs) to int8. Combining these two techniques allows us to fine-tune PLMs on a single consumer GPU such as RTX 4090.

GitHub: https://github.com/bofenghuang/vigogne

## WizardLM

> An Instruction-following LLM Using Evol-Instruct. Empowering Large Pre-Trained Language Models to Follow Complex Instructions

- GitHub: [nlpxucan/WizardLM: WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions (github.com)](https://github.com/nlpxucan/WizardLM)
- Review: [WizardLM: Evolving Instruction Datasets to Create a Better Model — YouTube](https://www.youtube.com/watch?v=5IAxCL4dHWk)

# BLOOM (BigScience)

> BigScience Large Open-science Open-access Multilingual Language Model.

- Hugging Face: [bigscience/bloom · Hugging Face](https://huggingface.co/bigscience/bloom)
- Hugging Face Demo: [Bloom Demo — a Hugging Face Space by huggingface](https://huggingface.co/spaces/huggingface/bloom_demo)

Here is a list of reproductions of or based on the BLOOM project:

- BLOOM-LoRA | Petals

## BLOOM-LoRA

> Low-Rank adaptation for various Instruct-Tuning datasets.

- GitHub: [linhduongtuan/BLOOM-LORA: Due to restriction of LLaMA, we try to reimplement BLOOM-LoRA (much less restricted BLOOM license here https://huggingface.co/spaces/bigscience/license) using Alpaca-LoRA and Alpaca_data_cleaned.json (github.com)](https://github.com/linhduongtuan/BLOOM-LORA)

## Petals

> Generate text using distributed 176B-parameter [BLOOM](https://huggingface.co/bigscience/bloom) or [BLOOMZ](https://huggingface.co/bigscience/bloomz) and fine-tune them for your own tasks.

- GitHub: [bigscience-workshop/petals: 🌸 Run 100B+ language models at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading (github.com)](https://github.com/bigscience-workshop/petals)

# Cerebras-GPT (Cerebras)

> A Family of Open, Compute-efficient, Large Language Models. Cerebras open sources seven GPT-3 models from 111 million to 13 billion parameters. Trained using the Chinchilla formula, these models set new benchmarks for accuracy and compute efficiency.

- Website: [Cerebras-GPT: A Family of Open, Compute-efficient, Large Language Models — Cerebras](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
- Hugging Face: [cerebras (Cerebras) (huggingface.co)](https://huggingface.co/cerebras)
- Review: [Checking out the Cerebras-GPT family of models — YouTube](https://www.youtube.com/watch?v=9P3_Zw_1xpw)

# Flamingo (Google/Deepmind)

> Tackling multiple tasks with a single visual language model

- Website: [Tackling multiple tasks with a single visual language model](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)

Here is a list of reproductions of or based on the Flamingo project:

- Flamingo — Pytorch | OpenFlamingo

## Flamingo — Pytorch

> Implementation of [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model), state-of-the-art few-shot visual question answering attention net, in Pytorch. It will include the perceiver resampler (including the scheme where the learned queries contributes keys / values to be attended to, in addition to media embeddings), the specialized masked cross attention blocks, and finally the tanh gating at the ends of the cross attention + corresponding feedforward blocks.

- GitHub: https://github.com/lucidrains/flamingo-pytorch

## OpenFlamingo

> Welcome to our open source version of DeepMind’s Flamingo model! In this repository, we provide a PyTorch implementation for training and evaluating OpenFlamingo models. We also provide an initial OpenFlamingo 9B model trained on a new Multimodal C4 dataset (coming soon). Please refer to our blog post for more details.

- GitHub: [mlfoundations/open_flamingo: An open-source framework for training large multimodal models (github.com)](https://github.com/mlfoundations/open_flamingo)

# FLAN (Google)

> This repository contains code to generate instruction tuning dataset collections. The first is the original Flan 2021, documented in [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652), and the second is the expanded version, called the Flan Collection, described in [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688) and used to produce [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) and [Flan-PaLM](https://arxiv.org/abs/2210.11416).

- GitHub: [google-research/FLAN (github.com)](https://github.com/google-research/FLAN)

Here is a list of reproductions of or based on the FLAN project:

- FastChat-T5 | Flan-Alpaca | Flan-UL2

## FastChat-T5

> We are excited to release FastChat-T5: our compact and commercial-friendly chatbot! that is Fine-tuned from Flan-T5, ready for commercial usage! and Outperforms Dolly-V2 with 4x fewer parameters.

- GitHub: [lm-sys/FastChat: The release repo for “Vicuna: An Open Chatbot Impressing GPT-4” (github.com)](https://github.com/lm-sys/FastChat#FastChat-T5)
- Hugging Face: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/huggingface_api.py

## Flan-Alpaca

> Instruction Tuning from Humans and Machines. This repository contains code for extending the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) synthetic instruction tuning to existing instruction-tuned models such as [Flan-T5](https://arxiv.org/abs/2210.11416). The pretrained models and demos are available on HuggingFace

- GitHub: [declare-lab/flan-alpaca: This repository contains code for extending the Stanford Alpaca synthetic instruction tuning to existing instruction-tuned models such as Flan-T5. (github.com)](https://github.com/declare-lab/flan-alpaca)

## Flan-UL2

> Flan-UL2 is an encoder decoder model based on the `T5` architecture. It uses the same configuration as the `UL2 model` released earlier last year. It was fine tuned using the "Flan" prompt tuning and dataset collection.

- Hugging Face: [google/flan-ul2 · Hugging Face](https://huggingface.co/google/flan-ul2)
- Review: [Trying Out Flan 20B with UL2 — Working in Colab with 8Bit Inference — YouTube](https://www.youtube.com/watch?v=cMT3RzjawEc)

# GALACTICA (Meta)

> Following [Mitchell et al. (2018)](https://arxiv.org/abs/1810.03993), this model card provides information about the GALACTICA model, how it was trained, and the intended use cases. Full details about how the model was trained and evaluated can be found in the [release paper](https://galactica.org/paper.pdf).

- GitHub: [galai/model_card.md at main · paperswithcode/galai (github.com)](https://github.com/paperswithcode/galai/blob/main/docs/model_card.md)

Here is a list of reproductions of or based on the GALACTICA project:

- Galpaca

## Galpaca

> GALACTICA 30B fine-tuned on the Alpaca dataset.

- Hugging Face: [GeorgiaTechResearchInstitute/galpaca-30b · Hugging Face](https://huggingface.co/GeorgiaTechResearchInstitute/galpaca-30b)
- Hugging Face: [TheBloke/galpaca-30B-GPTQ-4bit-128g · Hugging Face](https://huggingface.co/TheBloke/galpaca-30B-GPTQ-4bit-128g)

# GLM (General Language Model)

> GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.

Here is a list of reproductions of or based on the GLM project:

- ChatGLM-6B

## ChatGLM-6B

> ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level).
>
> ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue. The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning, feedback bootstrap, and reinforcement learning wit human feedback. With only about 6.2 billion parameters, the model is able to generate answers that are in line with human preference.

- GitHub: [THUDM/ChatGLM-6B: ChatGLM-6B：开源双语对话语言模型 | An Open Bilingual Dialogue Language Model (github.com)](https://github.com/THUDM/ChatGLM-6B)

# GPT-J (EleutherAI)

> **GPT-J** is an open source [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence) [language model](https://en.wikipedia.org/wiki/Language_model) developed by [EleutherAI](https://en.wikipedia.org/wiki/EleutherAI).[[1\]](https://en.wikipedia.org/wiki/GPT-J#cite_note-1) GPT-J performs very similarly to [OpenAI](https://en.wikipedia.org/wiki/OpenAI)’s [GPT-3](https://en.wikipedia.org/wiki/GPT-3) on various zero-shot down-streaming tasks and can even outperform it on code generation tasks.[[2\]](https://en.wikipedia.org/wiki/GPT-J#cite_note-2) The newest version, GPT-J-6B is a language model based on a data set called [The Pile](https://en.wikipedia.org/wiki/The_Pile_(dataset)).[[3\]](https://en.wikipedia.org/wiki/GPT-J#cite_note-3) The Pile is an open-source 825 [gibibyte](https://en.wikipedia.org/wiki/Byte#Multiple-byte_units) language modelling data set that is split into 22 smaller datasets.[[4\]](https://en.wikipedia.org/wiki/GPT-J#cite_note-4) GPT-J is similar to [ChatGPT](https://en.wikipedia.org/wiki/ChatGPT) in ability, although it does not function as a chat bot, only as a text predictor.[[5\]](https://en.wikipedia.org/wiki/GPT-J#cite_note-5)

- GitHub: https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b
- Demo: https://6b.eleuther.ai/

Here is a list of reproductions of or based on the GPT-J project:

- Dolly | GPT-J-6B instruction-tuned on Alpaca-GPT4

## Dolly (Databricks)

> Databricks’ Dolly, a large language model trained on the [Databricks Machine Learning Platform](https://www.databricks.com/product/machine-learning), demonstrates that a two-years-old open source model ([GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B)) can, when subjected to just 30 minutes of fine tuning on a focused corpus of 50k records ([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)), exhibit surprisingly high quality instruction following behavior not characteristic of the foundation model on which it is based. We believe this finding is important because it demonstrates that the ability to create powerful artificial intelligence technologies is vastly more accessible than previously realized.

- GitHub: [databrickslabs/dolly: Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform (github.com)](https://github.com/databrickslabs/dolly)
- Review: [Meet Dolly the new Alpaca model — YouTube](https://www.youtube.com/watch?v=AWAo4iyNWGc)

## GPT-J-6B instruction-tuned on Alpaca-GPT4

> This model was finetuned on GPT-4 generations of the Alpaca prompts, using LoRA for 30.000 steps (batch size of 128), taking over 7 hours in four V100S.

- Hugging Face: [vicgalle/gpt-j-6B-alpaca-gpt4 · Hugging Face](https://huggingface.co/vicgalle/gpt-j-6B-alpaca-gpt4?text=My+name+is+Teven+and+I+am)

## GPT4All-J

> Demo, data, and code to train open-source assistant-style large language model based on GPT-J

- GitHub: [nomic-ai/gpt4all: gpt4all: an ecosystem of open-source chatbots trained on a massive collections of clean assistant data including code, stories and dialogue (github.com)](https://github.com/nomic-ai/gpt4all)
- Review: [GPT4ALLv2: The Improvements and Drawbacks You Need to Know! — YouTube](https://www.youtube.com/watch?v=5icWiTvDQS0)

# GPT-NeoX (EleutherAI)

> This repository records [EleutherAI](https://www.eleuther.ai/)’s library for training large-scale language models on GPUs. Our current framework is based on NVIDIA’s [Megatron Language Model](https://github.com/NVIDIA/Megatron-LM) and has been augmented with techniques from [DeepSpeed](https://www.deepspeed.ai/) as well as some novel optimizations. We aim to make this repo a centralized and accessible place to gather techniques for training large-scale autoregressive language models, and accelerate research into large-scale training.

- GitHub: [EleutherAI/gpt-neox: An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library. (github.com)](https://github.com/EleutherAI/gpt-neox)

# h2oGPT (h2o.ai)

> Our goal is to make the world’s best open source GPT!

- GitHub: [h2oai/h2ogpt: Come join the movement to make the world’s best open source GPT led by H2O.ai (github.com)](https://github.com/h2oai/h2ogpt)
- Hugging Face: [H2ogpt Oasst1 256 6.9b App — a Hugging Face Space by h2oai](https://huggingface.co/spaces/h2oai/h2ogpt-oasst1-256-6.9b-hosted)

# HuggingGPT (Microsoft)

> HuggingGPT is a collaborative system that consists of an LLM as the controller and numerous expert models as collaborative executors (from HuggingFace Hub).

- GitHub: [microsoft/JARVIS: JARVIS, a system to connect LLMs with ML community (github.com)](https://github.com/microsoft/JARVIS)

# MPT-7B (Mosaic ML)

> MPT-7B is a GPT-style model, and the first in the MosaicML Foundation Series of models. Trained on 1T tokens of a MosaicML-curated dataset, MPT-7B is open-source, commercially usable, and equivalent to LLaMa 7B on evaluation metrics. The MPT architecture contains all the latest techniques on LLM modeling — Flash Attention for efficiency, Alibi for context length extrapolation, and stability improvements to mitigate loss spikes. The base model and several variants, including a 64K context length fine-tuned model (!!) are all available.

- Website: [Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs (mosaicml.com)](https://www.mosaicml.com/blog/mpt-7b)
- GitHub: [mosaicml/llm-foundry (github.com)](https://github.com/mosaicml/llm-foundry#mpt)
- Review: [MPT-7B — The First Commercially Usable Fully Trained LLaMa Model — YouTube](https://www.youtube.com/watch?v=NY0bLFqkBL0)

# NeMo — GPT-2B-001 (Nvidia)

> GPT-2B-001 is a transformer-based language model. GPT refers to a class of transformer decoder-only models similar to GPT-2 and 3 while 2B refers to the total trainable parameter count (2 Billion) [1, 2]. This model was trained on 1.1T tokens with [NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html).

- Hugging Face: https://huggingface.co/nvidia/GPT-2B-001

# OpenAssistant Models

> Conversational AI for everyone.

- Website: [Open Assistant (open-assistant.io)](https://open-assistant.io/)
- GitHub: [LAION-AI/Open-Assistant: OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so. (github.com)](https://github.com/LAION-AI/Open-Assistant)
- Hugging Face: [OpenAssistant (OpenAssistant) (huggingface.co)](https://huggingface.co/OpenAssistant)

# OpenLLaMA

> In this repo, we release a permissively licensed open source reproduction of Meta AI’s [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) large language model. In this release, we’re releasing a public preview of the 7B OpenLLaMA model that has been trained with 200 billion tokens. We provide PyTorch and Jax weights of pre-trained OpenLLaMA models, as well as evaluation results and comparison against the original LLaMA models. Stay tuned for our updates.

- GitHub: [openlm-research/open_llama (github.com)](https://github.com/openlm-research/open_llama)

# PaLM (Google)

> PaLM demonstrates the first large-scale use of the Pathways system to scale training to 6144 chips, the largest TPU-based system configuration used for training to date. The training is scaled using [data parallelism](https://en.wikipedia.org/wiki/Data_parallelism) at the Pod level across two [Cloud TPU v4 Pods](https://cloud.google.com/blog/topics/tpus/google-showcases-cloud-tpu-v4-pods-for-large-model-training), while using standard data and model parallelism within each Pod. This is a significant increase in scale compared to most previous LLMs, which were either trained on a single TPU v3 Pod (e.g., [GLaM](https://arxiv.org/abs/2112.06905), [LaMDA](https://arxiv.org/abs/2201.08239)), used pipeline parallelism to scale to 2240 A100 GPUs across GPU clusters ([Megatron-Turing NLG](https://arxiv.org/abs/2201.11990.pdf)) or used multiple TPU v3 Pods ([Gopher](https://arxiv.org/abs/2112.11446)) with a maximum scale of 4096 TPU v3 chips.

- Website: [Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance — Google AI Blog (googleblog.com)](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)

Here is a list of reproductions of or based on the PaLM project:

- PaLM (Concept of Mind)

## PaLM (Concept of Mind)

Introducing three new open-source PaLM models trained at a context length of 8k on C4. Open-sourcing LLMs is a necessity for the fair and equitable democratization of AI. The models of sizes 150m, 410m, and 1b are available to download and use here.

- GitHub: [conceptofmind/PaLM: An open-source implementation of Google’s PaLM models (github.com)](https://github.com/conceptofmind/PaLM)

# Palmyra Base 5B (Writer)

> Palmyra Base was primarily pre-trained with English text. Note that there is still a trace amount of non-English data present within the training corpus that was accessed through CommonCrawl. A causal language modeling (CLM) objective was utilized during the process of the model’s pretraining. Similar to GPT-3, Palmyra Base is a member of the same family of models that only contain a decoder. As a result, it was pre-trained utilizing the objective of self-supervised causal language modeling. Palmyra Base uses the prompts and general experimental setup from GPT-3 in order to conduct its evaluation per GPT-3.

- Hugging Face: [Writer/palmyra-base · Hugging Face](https://huggingface.co/Writer/palmyra-base)

Here is a list of reproductions of or based on the Palmyra Base project:

- Camel 5B

## Camel 🐪 5B

> Introducing Camel-5b, a state-of-the-art instruction-following large language model designed to deliver exceptional performance and versatility. Derived from the foundational architecture of [Palmyra-Base](https://huggingface.co/Writer/palmyra-base), Camel-5b is specifically tailored to address the growing demand for advanced natural language processing and comprehension capabilities.

- Hugging Face: [Writer/camel-5b-hf · Hugging Face](https://huggingface.co/Writer/camel-5b-hf)

# Polyglot (EleutherAI)

> Large Language Models of Well-balanced Competence in Multi-languages. Various multilingual models such as mBERT, BLOOM, and XGLM have been released. Therefore, someone might ask, “why do we need to make multilingual models again?” Before answering the question, we would like to ask, “Why do people around the world make monolingual models in their language even though there are already many multilingual models?” We would like to point out there is a dissatisfaction with the non-English language performance of the current multilingual models as one of the most significant reason. So we want to make multilingual models with higher non-English language performance. This is the reason we need to make multilingual models again and why we name them ‘Polyglot’.

- GitHub: [EleutherAI/polyglot: Polyglot: Large Language Models of Well-balanced Competence in Multi-languages (github.com)](https://github.com/EleutherAI/polyglot)

# Pythia (EleutherAI)

> Interpreting Autoregressive Transformers Across Time and Scale

- GitHub: [EleutherAI/pythia (github.com)](https://github.com/EleutherAI/pythia)

Here is a list of reproductions of or based on the Pythia project:

- Dolly 2.0

## Dolly 2.0 (Databricks)

> Dolly 2.0 is a 12B parameter language model based on the [EleutherAI](https://www.eleuther.ai/) [pythia](https://arxiv.org/abs/2304.01373) model family and fine-tuned exclusively on a new, high-quality human generated instruction following dataset, crowdsourced among Databricks employees.

- Website: [Free Dolly: Introducing the World’s First Open and Commercially Viable Instruction-Tuned LLM — The Databricks Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- Hugging Face: [databricks (Databricks) (huggingface.co)](https://huggingface.co/databricks)
- GutHub: [dolly/data at master · databrickslabs/dolly (github.com)](https://github.com/databrickslabs/dolly/tree/master/data)
- Review: [Dolly 2.0 by Databricks: Open for Business but is it Ready to Impress! — YouTube](https://www.youtube.com/watch?v=grEp5jipOtg)

# RedPajama-INCITE 3B and 7B (Together)

> The first models trained on the RedPajama base dataset: a 3 billion and a 7B parameter base model that aims to replicate the LLaMA recipe as closely as possible. In addition, we are releasing fully open-source instruction-tuned and chat models.

- Website: [Releasing 3B and 7B RedPajama-INCITE family of models including base, instruction-tuned & chat models — TOGETHER](https://www.together.xyz/blog/redpajama-models-v1)
- Hugging Face: [togethercomputer/RedPajama-INCITE-Base-3B-v1 · Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1), [togethercomputer/RedPajama-INCITE-Chat-3B-v1 · Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1), and [togethercomputer/RedPajama-INCITE-Instruct-3B-v1 · Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1)
- Hugging Face: [togethercomputer/RedPajama-INCITE-Base-7B-v0.1 · Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1), [togethercomputer/RedPajama-INCITE-Chat-7B-v0.1 · Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1), and [togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1 · Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1)

# Replit-Code (Replit)

> `replit-code-v1-3b` is a 2.7B Causal Language Model focused on Code Completion. The model has been trained on a subset of the [Stack Dedup v1.2 dataset](https://arxiv.org/abs/2211.15533). The training mixture includes 20 different languages, listed here in descending order of number of tokens:
> `Markdown`, `Java`, `JavaScript`, `Python`, `TypeScript`, `PHP`, `SQL`, `JSX`, `reStructuredText`, `Rust`, `C`, `CSS`, `Go`, `C++`, `HTML`, `Vue`, `Ruby`, `Jupyter Notebook`, `R`, `Shell`
> In total, the training dataset contains 175B tokens, which were repeated over 3 epochs -- in total, `replit-code-v1-3b` has been trained on 525B tokens (~195 tokens per parameter).

- Hugging Face: https://huggingface.co/replit/replit-code-v1-3b

# The RWKV Language Model

> RWKV: Parallelizable RNN with Transformer-level LLM Performance (pronounced as “RwaKuv”, from 4 major params: R W K V)

- GitHub: [BlinkDL](https://github.com/BlinkDL)/[RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- ChatRWKV: with “stream” and “split” strategies and INT8. 3G VRAM is enough to run RWKV 14B :) https://github.com/BlinkDL/ChatRWKV
- Hugging Face Demo: [HuggingFace Gradio demo (14B ctx8192)](https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio)
- Hugging Face Demo: [Raven (7B finetuned on Alpaca) Demo](https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B)
- RWKV pip package: https://pypi.org/project/rwkv/
- Review: [Raven — RWKV-7B RNN’s LLM Strikes Back — YouTube](https://www.youtube.com/watch?v=B3Qa2rRsaXo)

# Segment Anything (Meta)

> The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

- Website: [Introducing Segment Anything: Working toward the first foundation model for image segmentation (facebook.com)](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/?utm_source=twitter&utm_medium=organic_social&utm_campaign=segmentanything&utm_content=gif)
- GitHub: [facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. (github.com)](https://github.com/facebookresearch/segment-anything)

# StableLM (StabilityAI)

> A new open-source language model, [StableLM](https://github.com/stability-AI/stableLM/). The Alpha version of the model is available in 3 billion and 7 billion parameters, with 15 billion to 65 billion parameter models to follow. Developers can freely inspect, use, and adapt our StableLM base models for commercial or research purposes, subject to the terms of the CC BY-SA-4.0 license. StableLM is trained on a new experimental dataset built on The Pile, but three times larger with 1.5 trillion tokens of content. We will release details on the dataset in due course. The richness of this dataset gives StableLM surprisingly high performance in conversational and coding tasks, despite its small size of 3 to 7 billion parameters (by comparison, GPT-3 has 175 billion parameters)

- Website: [Stability AI Launches the First of its StableLM Suite of Language Models — Stability AI](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models)
- GitHub: [Stability-AI/StableLM: StableLM: Stability AI Language Models (github.com)](https://github.com/stability-AI/stableLM/)
- Hugging Face: [Stablelm Tuned Alpha Chat — a Hugging Face Space by stabilityai](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat)
- Review: [Stable LM 3B — The new tiny kid on the block. — YouTube](https://www.youtube.com/watch?v=0uI7SoMn0Es)

# StartCoder (BigCode)

> BigCode is an open scientific collaboration working on responsible training of large language models for coding applications. You can find more information on the main [website](https://www.bigcode-project.org/) or follow Big Code on [Twitter](https://twitter.com/BigCodeProject). In this organization you can find the artefacts of this collaboration: StarCoder, a state-of-the-art language model for code, The Stack, the largest available pretraining dataset with perimssive code, and SantaCoder, a 1.1B parameter model for code.

- Website: https://huggingface.co/bigcode
- Hugging Face: https://huggingface.co/spaces/bigcode/bigcode-editor and https://huggingface.co/spaces/bigcode/bigcode-playground
- Review: [Testing Starcoder for Reasoning with PAL — YouTube](https://www.youtube.com/watch?v=fXWYMdR2Dg0)

# XGLM (Meta)

> The XGLM model was proposed in [Few-shot Learning with Multilingual Language Models](https://arxiv.org/abs/2112.10668).

- GitHub: https://github.com/facebookresearch/fairseq/tree/main/examples/xglm
- Hugging Face: https://huggingface.co/docs/transformers/model_doc/xglm

# Other Repositories

## A’eala

- Hugging Face: [Aeala (A’eala) (huggingface.co)](https://huggingface.co/Aeala)

## chavinlo

- Hugging Face: [chavinlo (Chavez) (huggingface.co)](https://huggingface.co/chavinlo)

## chainyo

- Hugging Face: [chainyo (Thomas Chaigneau) (huggingface.co)](https://huggingface.co/chainyo)

## couchpotato888

- Hugging Face: [couchpotato888 (Phil Wee) (huggingface.co)](https://huggingface.co/couchpotato888)

## crumb

- Hugging Face: https://huggingface.co/crumb

## **digitous**

- Hugging Face: [digitous (Erik) (huggingface.co)](https://huggingface.co/digitous)

## eachadea

- Hugging Face: [eachadea (eachadea) (huggingface.co)](https://huggingface.co/eachadea)

## **Knut Jägersberg**

- Hugging Face: https://huggingface.co/KnutJaegersberg

## KoboldAI

- Hugging Face: [KoboldAI (KoboldAI) (huggingface.co)](https://huggingface.co/KoboldAI)

## LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions

> LaMini-LM is a collection of small-sized, efficient language models distilled from ChatGPT and trained on a large-scale dataset of 2.58M instructions. We explore different model architectures, sizes, and checkpoints, and extensively evaluate their performance across various NLP benchmarks and through human evaluation.

- Paper: [[2304.14402\] LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions (arxiv.org)](https://arxiv.org/abs/2304.14402)
- GitHub: [mbzuai-nlp/LaMini-LM: LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions (github.com)](https://github.com/mbzuai-nlp/LaMini-LM)
- Review: [LaMini-LM — Mini Models Maxi Data! — YouTube](https://www.youtube.com/watch?v=TeJrG3juAL4&t=42s)

## Teknium

- Hugging Face: https://huggingface.co/teknium

**I hope you have enjoyed this article. If you have any questions or comments, please provide them here.**

# List of all Foundation Models

Sourced from: [A List of 1 Billion+ Parameter LLMs (matt-rickard.com)](https://matt-rickard.com/a-list-of-1-billion-parameter-llms)

- GPT-J (6B) (EleutherAI)
- GPT-Neo (1.3B, 2.7B, 20B) (EleutherAI)
- Pythia (1B, 1.4B, 2.8B, 6.9B, 12B) (EleutherAI)
- Polyglot (1.3B, 3.8B, 5.8B) (EleutherAI)
- J1/Jurassic-1 (7.5B, 17B, 178B) (AI21)
- J2/Jurassic-2 (Large, Grande, and Jumbo) (AI21)
- LLaMa (7B, 13B, 33B, 65B) (Meta)
- OPT (1.3B, 2.7B, 13B, 30B, 66B, 175B) (Meta)
- Fairseq (1.3B, 2.7B, 6.7B, 13B) (Meta)
- GLM-130B YaLM (100B) (Yandex)
- YaLM (100B) (Yandex)
- UL2 20B (Google)
- PanGu-α (200B) (Huawei)
- Cohere (Medium, XLarge)
- Claude (instant-v1.0, v1.2) (Anthropic)
- CodeGen (2B, 6B, 16B) (Salesforce)
- NeMo (1.3B, 5B, 20B) (NVIDIA)
- RWKV (14B)
- BLOOM (1B, 3B, 7B)
- GPT-4 (OpenAI)
- GPT-3.5 (OpenAI)
- GPT-3 (ada, babbage, curie, davinci) (OpenAI)
- Codex (cushman, davinci) (OpenAI)
- T5 (11B) (Google)
- CPM-Bee (10B)
- Cerebras-GPT

# Resources

- PRIMO.ai Large Language Model (LLM): https://primo.ai/index.php?title=Large_Language_Model_(LLM)
- A Survey of Large Language Models: [[2303.18223\] A Survey of Large Language Models (arxiv.org)](https://arxiv.org/abs/2303.18223)

![img](https://miro.medium.com/v2/resize:fit:1400/1*HpnTv6oEvNTSIHAZ4uQwJg.png)

[ A Survey of Large Language Models (arxiv.org)](https://arxiv.org/abs/2303.18223) — Page 5

- LLMMaps — A Visual Metaphor for Stratified Evaluation of Large Language Models: https://arxiv.org/abs/2304.00457

![img](https://miro.medium.com/v2/resize:fit:1400/1*VxwEWou6umGCNj9Arl6IaQ.png)

https://arxiv.org/pdf/2304.00457.pdf — Page 7

- A brief history of LLaMA models ([A brief history of LLaMA models — AGI Sphere (agi-sphere.com)](https://agi-sphere.com/llama-models/))
- Google “We Have No Moat, And Neither Does OpenAI” (https://www.semianalysis.com/p/google-we-have-no-moat-and-neither)
- Chatbot Arena ([Chat with Open Large Language Models (lmsys.org)](https://chat.lmsys.org/?arena))
- [Ahead of AI #8: The Latest Open Source LLMs and Datasets](https://magazine.sebastianraschka.com/p/ahead-of-ai-8-the-latest-open-source)
- Open LLM Leaderboard ([Open LLM Leaderboard — a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard))
