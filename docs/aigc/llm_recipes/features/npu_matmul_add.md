# matmul_add融合优化

## 问题分析
模型训练中开启了梯度累加功能，但累加效率较慢，梯度累加中的 Add 算子占比较高。

## 解决方法
MindSpeed将matmul操作和add操作合并成一个融合算子。算子接口见[link](../ops/npu_matmul_add.md)。

## 使用场景
llama、gpt大模型均使用。

## 使用方法
先安装CANN-NNAL并初始化添加环境，例如：
CANN-NNAL默认安装路径
source /usr/local/Ascend/nnal/atb/set_env.sh

去掉`--no-gradient-accumulation-fusion`即可调用npu_matmul_add_fp32融合算子。

## 使用效果
开启融合算子，llama2_70B_4k_tp2_pp2_vpp1_dp2性能可提升1.5%。
