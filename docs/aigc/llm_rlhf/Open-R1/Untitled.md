# simple-rl-reason

这个笔记记录复现[simple-rl-reason](https://hkust-nlp.notion.site/simplerl-reason)工作的主要结论，该工作有几个关键点：

1. 采用跟deepseek一样的策略，训练一个SimpleRL-zero版本和一个SimpleRL版本；
2. 仅仅采用7B模型和8K的数据集，就复现了long COT和自我学习的涌现现象；并且在benchmark上性能有显著的提升；
3. 该工作还复现了response length随着训练步长增加的现象；（reward也随之提升！）

## baseline模型

1. 实验采用的baseline模型为：[Qwen2.5-Math-7B-Base](https://huggingface.co/Qwen/Qwen2.5-Math-7B)
2. 据HW那边介绍，如果采用`Qwen2.5-Math-7B-Instruct`模型训练，那么实验中不会出现response length增加的现象！

## 强化学习算法

1. 原文采用的是PPO算法；
2. 复现计划采用GRPO算法，基于Mindspeed-LLM仓库；

## benchmark

1. 评测可以参考这个目录：/root/llm_workspace/projects/simpleRL-reason/eval
2. 评估过程禁用了调用代码解决问题；

模型测试的基准有：
1. AIME2024: 一个面向高中生的数学竞赛，AIME2024指的是2024年的比赛题目集，主要用于评估模型解决高级数学问题的能力。
2. AMC23:AMC（American Mathematics Competitions）系列考试之一，AMC23可能是指AMC 10或AMC 12在2023年的版本，这些考试旨在识别具有卓越数学才能的学生，并测试模型对中学数学概念的理解和应用能力。
3. GSM8K:一个包含大约8000个小学级别的数学问题的数据集，设计用来评估语言模型在基础算术和简单应用题上的表现。
4. MATH-500:从更大的MATH数据集中精选出的500个具有挑战性的数学问题，专注于评估大型语言模型在解决复杂数学问题时的表现。
5. Minerva Math:由谷歌开发，专门用于训练和评估其Minerva模型在数学推理任务上的性能，该数据集包含了各种难度级别的数学问题，强调逻辑推理和数学知识的应用。
6. OlympiadBench:汇集了国际数学奥林匹克竞赛及其他类似高水平竞赛题目的基准测试集，旨在评估模型在处理最具挑战性的数学问题时的能力，这些问题通常需要创造性和深刻的理论背景。

## 数据集

1. 强化学习的数据集为MATH-8K，存放位置：`/root/llm_workspace/projects/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt.json`，一共8523个样本；
2. 用于做Long COT的SFT数据集，暂时没有发布；

## 训练两阶段

1. SimpleRL-zero基于`Qwen2.5-MATH-7B`直接做强化学习，没有做`SFT`；
2. SimpleRL训练分为两个阶段：
   1. 做一个long COT的SFT：采用`QwQ-32B-Preview`模型在`MATH-8K`上蒸馏出来的COT;
   2. 再基于MATH-8K数据集做强化学习训练；

## reward设计

基于规则的reward设计：

1. 如果模型输出的答案的格式是对的，并且和真实答案一致，则reward=1；
2. 如果模型输出的答案的格式是对的，但是和真实答案不一致，则reward=-0.5；
3. 如果模型输出的答案的格式不对，则reward=-1；



# Codes

这个文档记录代码相关的内容。假定当前目录为`/root/llm_workspace/projects/MindSpeed-LLM`

## 权重转换

当前下载的权重为`huggingface`格式，需要转换为`mindspeed-LLM`里面的`mcore`格式

1. `Qwen2.5-MATH-7B`存放位置：`/root/llm_workspace/models/Qwen/Qwen2.5-Math-7B`
2. 官方提供的转换脚本：`examples/mcore/qwen25_math/ckpt_convert_qwen25_math_hf2mcore.sh`
3. 修改脚本中的路径和`TP`，`PP`的值；修改脚本保存到当前目录：`simpleRL/ckpt_convert_qwen25_math_hf2mcore.sh`
4. 在`mindspeed-LLM`目录下执行：`bash simpleRL/ckpt_convert_qwen25_math_hf2mcore.sh`
5. 生成的权重文件保存为：`/root/llm_workspace/models/Qwen/Qwen2.5-Math-7B/mcore_tp4`

## 数据集转换

1. 原始数据集为MATH-8K，存放位置为：/root/llm_workspace/projects/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt.json；
2. 数据集存放脚本为：/root/llm_workspace/projects/MindSpeed-LLM/preprocess_data.py;
3. 观察MATH-8K数据集，发现他们符合alpaca格式，转换方法可以参考这个[文档](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/alpaca_dataset.md);
4. 因此可以调用preprocess_data.py进行转换，转换命令为：

   ```bash
    python preprocess_data.py \
    --input /root/llm_workspace/projects/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt.json \
    --tokenizer-name-or-path /root/llm_workspace/models/Qwen/Qwen2.5-Math-7B \
    --output-prefix simpleRL/dataset/math_8k \
    --workers 8 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type qwen_math_r1 \
    --map-keys '{"prompt":"input","query":"","response":"gt_answer"}' # 这里注意转换key
   ```

转化后的数据集描述：
1. 包含关键字：'input_ids', 'attention_mask', 'labels'
2. 'input_ids': list数据，包含了prompt和response两部分tokenze之后的ids
3. 'attention_mask': list数据，全为1，长度跟input_ids一样
4. 'labels': list数据，长度跟input_ids一致，前面prompt部分填充为-100，后面response部分为真实tokenzied之后的值；

## 代码检查

调试过程中发现的问题：

1. 调试过程好像发现推理采用了12张卡？
2. 默认训练使用的reward，包含了`acc`和`format`，但是`format`里面是对`r"^<think>.*?</think>\s*<answer>.*?</answer>$"`这种模式做匹配，这个可能不符合`simpleRL`的逻辑，当前数据集采用的是Qwen的prompt type，查看`simpleRL`源码，发现并没有包含`format`这个reward，看起来像是在`acc`里面实现了的；
3. 发现Qwen系列模型的`max_position_embeddings`参数规律：
   1. 如果是MATH系列模型，`max_position_embeddings=4096`
   2. 如果是通用系列模型，`max_position_embeddings=32768`
   由此，需要重新检查模型的配置参数，也就是文件`model/qwen25-7b.yaml`，也许需要采用`model/qwen25-math-7b.yaml`
4. 调试发现：`global_batch_size % data_parallel_size*micro_batch_size==0`，其中`train_gpu+infer_gpu)/model_parallel`，这里将infer_gpu和train_gpu加到一起，有点奇怪;
5. 发现模型的准备输入的时候，`input_ids += source_ids + target_ids`，为什么把target_ids也加到source_ids上面了呢？
    1. `mindspeed_llm/tasks/preprocess/data_handler.py`: 349行（目前考虑不修改数据预处理代码，而是修改GRPO的训练代码）
    2. 通过修改`mindspeed_llm/tasks/posttrain/rlxf/workers/actor_hybrid.py`212-219行，在进入采样阶段，把label（response）部分拿掉，这样可以进行正确的推理；
6. 打印模型的输入`input_ids`，并解码后，发现system prompt重复了一遍，已解决此bug；
    1. 转换`map_keys:--map-keys '{"prompt":"question","query":"","response":"gt_answer"}'`
    2. 修改`/root/llm_workspace/projects/MindSpeed-LLM/mindspeed_llm/tasks/preprocess/templates.py的157行`

## 模型训练

1. 初步实验，发现response length并没有随着训练步长增加而变长，reward也没有增加，说明当前训练存在问题，需要检查：
   1. 考虑调试模型训练的时候，数据集是否有问题；
   2. 发现当前默认的GRPO的reward跟simpleRL不一致，需要修改；
   3. 期望的acc和response曲线![simpleRL-zero](image.png)

## FAQ

1. 转换权重，报错：
    ```bash
      File "/root/llm_workspace/miniconda3/envs/mindspeed_llm/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py", line 121, in run_cuda_setup
    binary_name, cudart_path, cc, cuda_version_string = evaluate_cuda_setup()
    File "/root/llm_workspace/miniconda3/envs/mindspeed_llm/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py", line 344, in evaluate_cuda_setup
    ccs = get_compute_capabilities()
    File "/root/llm_workspace/miniconda3/envs/mindspeed_llm/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py", line 327, in get_compute_capabilities
    cc_major, cc_minor = torch.cuda.get_device_capability(torch.cuda.device(i))
    TypeError: get_device_capability() takes 0 positional arguments but 1 was given
    [ERROR] 2025-02-24-16:07:46 (PID:293686, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
    ```
    解决办法：卸载安装包-`bitsandbytes`，不用这个包也可以；

2. 启动训练，报错：
    ```bash
      File "/root/llm_workspace/miniconda3/envs/mindspeed_llm/lib/python3.10/site-packages/wandb/docker/auth.py", line 239, in load_config
    {"auths": cls.parse_auth(config_dict.pop("auths"), raise_on_error=True)}
    File "/root/llm_workspace/miniconda3/envs/mindspeed_llm/lib/python3.10/site-packages/wandb/docker/auth.py", line 190, in parse_auth
    username, password = decode_auth(entry["auth"])
    File "/root/llm_workspace/miniconda3/envs/mindspeed_llm/lib/python3.10/site-packages/wandb/docker/auth.py", line 380, in decode_auth
    login, pwd = s.split(b":", 1)
    ValueError: not enough values to unpack (expected 2, got 1)
    ```
    尝试办法：
        1. 设置环境变量：`export WANDB_API_KEY=xxx`
            2. 手动使用API登录：`wandb.login(key='xxx')`
           解决办法：
            将docker的配置文件去掉：`mv ~/.docker/config.json ~/.docker/config.json_bak`
        3.
