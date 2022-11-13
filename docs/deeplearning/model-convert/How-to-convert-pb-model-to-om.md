# 使用 ATC 工具 将 pb 模型转 om 格式

关于 ATC 工具的介绍 [ATC 工具使用指导 ](https://support.huaweicloud.com/ti-atc-A200_3000/altasatc_16_002.html)

### ATC 的使用
在环境准备好后， ATC 工具的使用比较简单:

```shell
atc --framework=3 --model=yolov3_coco.pb --output=yolov3 --soc_version=Ascend310 --input_shape="input:1,416,416,3"
```

|参数|使用帮助|
|:-:|:-:|
|framework | DL Framework type(0:Caffe; 1:MindSpore; 3:Tensorflow)|
| model    |         Model file   |
|output    |       Output file path&name(needn't suffix, will add .om automatically).|
|input_shape |       Shape of input data. Separate multiple nodes with semicolons (;).Use double quotation marks (") to enclose each argument.E.g.: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"|
|soc_version   |    The soc version. E.g.: "Ascend310"|

其他参数细节可以参考[华为官方文档](https://support.huaweicloud.com/ti-atc-A200_3000/altasatc_16_003.html)


### FAQ

我在使用 ATC 工具过程中，也遇到一些坑，通过摸索和求助华为工程师，大部分问题都得到解决，下面是我总结的使用 ATC 时需要注意一些细节：


- 模型 input 节点的 data_type， input_shape 怎么设定？
>客户根据自己需要的去冻结自己的PB模型，我们的 atc 是支持各种格式的转换的，input_shape 根据网络的实际输入大小来定， 例如对于 inceptionv3, input_shape=[N， 224， 224， 3]， 对于 Resnet50， input_shape=[N， 299， 299， 3].
>对于 data_type，根据实际需求， 可以设置 “int8, float32, string“ 等等

- 图像的预处理操作比如 Crop、Resize、Normalize 等 是否应该加到计算图中？

>不建议放，会增加模型的复杂度。
>
>我们昇腾有提供 DVPP硬件的预处理和 AIPP 软件处理的能力.
如果要使用DVPP就需要使用我们昇腾的ACL软件框架来处理


- 模型的输出节点是直接选原来的网络定义的节点还是要自己定义？
> 根据华为提供的 model zoo， 模型的输出节点一般都是网络定义好的， 在模型 freeze 成 pb 格式时就应该设置好.
>
> PB 到 om 我们支持指定 outnode 截断转换

- 模型转换过程中的 insert_op_conf 应该如何设置？

 在第一次将转换成功 resnet50 的 om 模型交给华为进行推理测试时失败，华为工程师发给我一个  AIPP  图像预处理算子， 再次转换之后的模型推理成功。

```sh
atc --framework=3 --model=resnet50.pb --output=resnet50 insert_op_conf=test.aipp.config   --soc_version=Ascend310 --input_shape="input:1,224,224,3"
```

```sh
aipp_op {
    aipp_mode: static
    related_input_rank: 0
    mean_chn_0: 104
    mean_chn_1: 117
    mean_chn_2: 123
    var_reci_chn_0: 1.0
    var_reci_chn_1: 1.0
    var_reci_chn_2: 1.0
    matrix_r0c0: 256
    matrix_r0c1: 454
    matrix_r0c2: 0
    matrix_r1c0: 256
    matrix_r1c1: -88
    matrix_r1c2: -183
    matrix_r2c0: 256
    matrix_r2c1: 0
    matrix_r2c2: 359
    input_bias_0: 0
    input_bias_1: 128
    input_bias_2: 128
    input_format: YUV400_U8
    csc_switch: true
    src_image_size_w: 224
    src_image_size_h: 224
    crop: true
}
```

AIPP（AI Preprocessing）用于在AI Core上完成图像预处理，包括色域转换（转换图像格式）、数据归一化（减均值/乘系数）和抠图（指定抠图起始点，抠出神经网络需要大小的图片. 关于 AIPP 详细文档可以参考 [AIPP 配置](https://support.huaweicloud.com/ti-atc-A200_3000/altasatc_16_007.html)
