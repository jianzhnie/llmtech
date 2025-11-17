## LLM RLHF Framework

- [大模型RL框架的演进与发展趋势](rlhf/infra/RL-Infra_overview.md)
- [面向 LLM 的开源强化学习库](rlhf/infra/Open-source-rl-library.md)
- [RLHF 训练框架 Slime](rlhf/infra/Slime.md)
- [RLHF 训练框架 ROLL](rlhf/infra/ROLL.md)
- [RLHF 中的 PPO 代码拆解](rlhf/infra/RLHF中的PPO代码拆解.md)
- [RLHF 训练框架 NeMo-Aligner](rlhf/infra/NeMo-Aligner.md)
- [RLHF 训练框架 DeepSpeedChat](rlhf/infra/DeepSpeedChat.md)
- [RLHF 训练框架 OpenR](rlhf/infra/OpenR.md)
- [RLHF 训练框架 AReaL](rlhf/infra/AReaL.md)
- [RLHF 训练框架 ARealLite](rlhf/infra/ARealLite.md)
- [RLHF 训练框架 AsyncFlow](rlhf/infra/AsyncFlow.md)
- [RLHF 训练框架 OpenRLHF](rlhf/infra/OpenRLHF.md)
- [RLHF 训练框架 OpenRLHF 源码解读](rlhf/infra/OpenRLHF源码解读.md)
- [RLHF 训练框架 VeRL](rlhf/infra/Verl.md)
- [RLHF 训练框架 VeRL 源码解读](rlhf/infra/Verl源码解读.md)
- [RLHF 训练框架 VeRL 参数配置指南](rlhf/infra/Verl参数配置.md)
- [OpenRLHF & &Verl参数转换指南](rlhf/infra/OpenRLHF&Verl参数转换指南.md)
- [从 Ray 角度分析 OpenRLHF 和 Verl 的工程设计](rlhf/infra/Ray_OpenRLHF_Verl.md)
- [Ray与LLM强化学习框架设计](rlhf/infra/Ray与LLM强化学习框架设计.md)


## Verl 源码分析

### 核心实现
- [核心算法实现](rlhf/infra/verl/core_algos.md)
- [Verl 单控制器设计详解](rlhf/infra/verl/verl.single_controller设计详解.md)
- [Verl 源码解析与 Hybrid Flow 编程范式](rlhf/infra/verl/verl_design.md)
- [Verl 中PPO 示例架构详解](rlhf/infra/verl/verl_ppo.md)

### Actor 实现
- [FSDP Actor 实现](rlhf/infra/verl/fsdp_actor.md)
- [FSDP Actor Worker](rlhf/infra/verl/fsdp_actor_worker.md)
- [Megatron Actor 实现](rlhf/infra/verl/megatron_actor.md)
- [FSDP Backend](rlhf/infra/verl/fsdp_backend.md)
- [Megatron Backend](rlhf/infra/verl/megatron_backend.md)

### Critic 实现
- [FSDP Critic 实现](rlhf/infra/verl/fsdp_critic.md)
- [FSDP Critic Worker](rlhf/infra/verl/fsdp_critic_worker.md)
- [Megatron Critic 实现](rlhf/infra/verl/megatron_critic.md)

### Rollout 相关
- [Hugging Face Rollout](rlhf/infra/verl/hf_rollout.md)
- [VLLM Rollout](rlhf/infra/verl/vllm_rollout.md)
- [Rollout Schemas](rlhf/infra/verl/rollout_schemas.md)

### VLLM 集成
- [FSDP VLLM 集成](rlhf/infra/verl/fsdp_vllm.md)
- [Megatron VLLM 集成](rlhf/infra/verl/megatron_vllm.md)
- [VLLM Server](rlhf/infra/verl/vllm_server.md)

### 奖励管理
- [朴素奖励管理器](rlhf/infra/verl/naive_reward_manager.md)



## LLM RLHF Intro

- [理解 RLHF](rlhf/intro/rlhf_advance.md)
- [Chip Huyen 对 RLHF 的分析](rlhf/intro/rlhf_chiphuyen.md)
- [RLHF 相关知识整理](rlhf/intro/rlhf_overview.md)
- [RLHF 中KL 散度的近似计算](rlhf/intro/KL散度的近似计算方法.md)
- [RLHF 中的 Policy Gradient Algorithms](rlhf/intro/rlhf_policy_gradient.md)
- [浅谈 GRPO 的系列改进（From GRPO to DAPO and GSPO）](rlhf/intro/grpo-to-dapo-and-gspo.md)
- [重新思考 PPO-Clip — GRPO 时代下的各种变体](rlhf/intro/ppo_clip.md)
- [截断重要性采样（TIS）](rlhf/intro/truncated_importance_sampling.md)
- [动态微调（Dynamic Fine-Tuning）](rlhf/intro/Dynamic-Fine-Tuning.markdown)


## LLM RLHF Algorithm and Paper


- [直接偏好优化 (DPO)](rlhf/paper/rlhf_dpo.md)
- [直接偏好优化 (DPO) 推导](rlhf/paper/rlhf_dpo_notes.md)
- [Kahneman-Tversky-Optimization (KTO)](rlhf/paper/rlhf_kto.md)
- [RLOO](rlhf/paper/RLOO.md)
- [DeepSeek-R1：通过强化学习激励 LLMs 的推理能力](rlhf/paper/DeepSeek-R1.md)
- [Kimi k1.5：使用 LLM 扩展强化学习](rlhf/paper/KimiK1.5.md)
- [DAPO: 一个开源的大规模 LLM 强化学习系统](rlhf/paper/DAPO.md)
- [深入理解 R1-Zero 类训练：一个批判性视角](rlhf/paper/DR.GRPO.md)
- [DeepScaleR：通过扩展强化学习超越 o1](rlhf/paper/deepscaler.md)
- [REINFORCE++：一种简单高效的大型语言模型对齐方法](rlhf/paper/REINFORCE++.md)
- [ChatGPT O1 Reasoning](rlhf/paper/chatgpt_O1.md)
- [过程奖励模型（Process Reward Model）](rlhf/paper/PRM.md)
- [数学推理中过程奖励模型的开发经验](rlhf/paper/PRM_Reasoning.md)
- [ReFT: 通过强化微调提升推理能力](rlhf/paper/ReFT.md)
- [拒绝采样（Reject Sampling）在 RLHF 中的应用](rlhf/paper/RejectSampling.md)
- [ReST-MCTS：通过过程奖励引导的树搜索实现 LLM 自训练](rlhf/paper/ReST-MCTS.md)
- [rStar-Math：小型语言模型通过自我进化的深度思考掌握数学推理](rlhf/paper/rStar-Math.md)
- [GRPO-λ (动态长度惩罚)](rlhf/paper/GRPO-lambda.md)
