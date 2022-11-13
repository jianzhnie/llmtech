# Tensorflow 手动修改计算图

有时，我们在没有源代码的情况下，想要修改别人构建好的计算图，可以通过修改GraphDef中的节点来实现。

### 1. 修改某一节点的输出
当我们想要重新构建某一节点后面的计算图时，可以通过修改该节点输出数据的流向来实现。

比如，我们想修改 ”resnet_v1_50/concat_7“ 这一节点后的计算图：
```python
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph("model.ckpt.meta", clear_devices=True)

    node = graph.get_tensor_by_name('resnet_v1_50/concat_7:0')
    with tf.name_scope('new_graph'):
        new_node = tf.add(node, tf.zeros_like(node), name='add')

```

### 2. 修改某一节点的输入
由于TensorFlow版本的差异，某些函数在不同版本中的定义可能略有不同。当某一函数的输入参数发生变化时，需要相应地在计算图中修改该节点输入，才可以在不同版本中使用。

比如，在TensorFlow 1.3中，GatherTree的定义为：
```python
def gather_tree(step_ids, parent_ids, sequence_length, name=None):
    '''
    Args:
        step_ids: A `Tensor`. Must be one of the following types: `int32`.
        `[max_time, batch_size, beam_width]`.
        parent_ids: A `Tensor`. Must have the same type as `step_ids`.
        `[max_time, batch_size, beam_width]`.
        sequence_length: A `Tensor`. Must have the same type as `step_ids`.
        `[batch_size, beam_width]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `step_ids`.
        `[max_time, batch_size, beam_width]`.
    '''
    # 具体定义省略

    return result
```
而在后面的版本中，GatherTree的定义为：
```python
def gather_tree(step_ids, parent_ids, max_sequence_lengths, end_token, name=None):
    '''
    Args:
        step_ids: A `Tensor`. Must be one of the following types: `int32`.
        `[max_time, batch_size, beam_width]`.
        parent_ids: A `Tensor`. Must have the same type as `step_ids`.
        `[max_time, batch_size, beam_width]`.
        max_sequence_lengths: A `Tensor` of type `int32`. `[batch_size]`.
        end_token: A `Tensor`. Must have the same type as `step_ids`. `[]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `step_ids`.
        `[max_time, batch_size, beam_width]`.
    '''
    # 具体定义省略

    return result
```
其中，输入参数共有两处变化：
- sequence_length 改为 max_sequence_lengths
- 增加了end_token

经查找，在TensorFlow 1.3生成的计算图中，sequence_length对应的节点为"Select_69"；而end_token需要手动加入。

```python
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph("model.ckpt.meta", clear_devices=True)

    sequence_length = graph.get_tensor_by_name('Select_69:0')
    max_sequence_length = tf.reduce_max(sequence_length, reduction_indices=[1], name='MaxSequenceLength')
    end_token = tf.constant(0, name='EndToken')

sess = tf.Session(graph=graph)

nodes = []
for node in sess.graph_def.node:
    if 'GatherTree' in node.name:
        print(node)
        for i, inp in enumerate(node.input):
            if 'Select' in inp:
                p = node.input.pop(i)
            node.input.append('MaxSequenceLength')
            node.input.append('EndToken')
        else:
            pass
        nodes.append(node)

mod_graph_def = tf.GraphDef()
mod_graph_def.node.extend(nodes)

tf.train.write_graph(mod_graph_def, './', 'new_model.pb', as_text=False)
```
这时，导出的.pb文件就可以在新版本的TensorFlow中导入并使用了。
