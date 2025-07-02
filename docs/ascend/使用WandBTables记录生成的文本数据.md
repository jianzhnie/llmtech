使用 WandB Tables 记录生成的文本数据

在训练或评估模型的过程中，我们通常需要记录模型的输出、预测结果或其他相关信息。WandB 的 `Table` 对象可以帮助我们以结构化的方式保存这些信息，方便后续可视化和分析。

本教程将介绍如何使用 `wandb.Table` 来记录和展示生成的文本或其他数据。

## 基本用法：创建一个简单的表格

我们可以直接定义表格的列（`columns`）和数据（`data`）来初始化一个 `Table`。

```python
import wandb

# 创建一个简单的 Table，包含三列：a, b, c
table = wandb.Table(
    columns=["a", "b", "c"],
    data=[
        ["1a", "1b", "1c"],
        ["2a", "2b", "2c"],
    ]
)
```

这里，`columns` 是列名列表，`data` 是二维数组，每一行对应一条记录。

------

## 更进一步：记录模型预测结果

假设你有一个模型在图像上进行了预测，并且你想记录每张图像的相关信息（如图像 ID、图片、预测标签和真实标签）。

```python
# 模拟数据
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 创建 Table
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

在这个例子中，`wandb.Image` 用来包装图片数据，这样图片会在 WandB 的 UI 上直观显示。

------

## 动态添加数据：逐行插入记录

在一些情况下，我们需要在模型推理过程中动态地将每一条数据插入到表格中，比如我们还想记录每一类别的置信度得分。

```python
# 定义列名
columns = ["id", "image", "guess", "truth"]
for digit in range(10):
    columns.append(f"score_{digit}")

# 初始化 Table
test_table = wandb.Table(columns=columns)

# 遍历测试数据集并逐行添加数据
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]
    guess_label = my_model.predict(img)

    # 假设模型返回每个类别的置信度分数
    confidence_scores = my_model.predict_proba(img)  # 返回一个长度为10的列表

    test_table.add_data(
        img_id,
        wandb.Image(img),
        guess_label,
        true_label,
        *confidence_scores  # 展开置信度分数
    )
```

- `add_data()` 用来添加一行数据。
- `*confidence_scores` 表示将列表中的元素分别作为独立参数传入。

------

## 小结

使用 `wandb.Table` 的核心步骤：

1. **定义列名**：清晰地定义好你要记录哪些字段。
2. **初始化表格**：可以一次性用 `data` 初始化，也可以使用 `add_data` 动态添加。
3. **使用`wandb.Image`**：如果有图像数据，可以用 `wandb.Image` 封装以便可视化。
4. **将 Table 记录到 WandB**：通常通过 `wandb.log({"my_table": table})` 将表格同步到服务器，方便查看。

------

## 示例：完整工作流程

```python
import wandb

wandb.init(project="your-project-name")

# 初始化 Table
columns = ["prompt", "generated_text", "score"]
table = wandb.Table(columns=columns)

# 假设你有多个 prompts 和对应的生成文本
prompts = ["Once upon a time", "In a galaxy far away"]
generated_texts = ["Once upon a time, there was a cat.", "In a galaxy far away, aliens danced."]
scores = [0.95, 0.89]  # 模型的生成质量得分

# 填充表格
for prompt, text, score in zip(prompts, generated_texts, scores):
    table.add_data(prompt, text, score)

# 记录到 wandb
wandb.log({"generated_text_table": table})
```

这样你就能在 WandB 界面上直观查看每个 prompt 和对应的生成文本及打分！
