

## Bag-of-words

Bag-of-words模型是信息检索领域常用的文档表示方法。在信息检索中，BOW模型假定对于一个文档，忽略它的单词顺序和语法、句法等要素，将其仅仅看作是若干个词汇的集合，文档中每个单词的出现都是独立的，不依赖于其它单词是否出现。也就是说，文档中任意一个位置出现的任何单词，都不受该文档语意影响而独立选择的。

词袋模型能够把一个句子转化为向量表示，是比较简单直白的一种方法，它不考虑句子中单词的顺序，只考虑词表（vocabulary）中单词在这个句子中的出现次数。下面直接来看一个例子吧（例子直接用wiki上的例子）：

### 例子

Wikipedia 上给出了如下例子:

> John likes to watch movies. Mary likes too.
>
> John also likes to watch football games.

根据上述两句话中出现的单词, 我们能构建出一个**字典** (dictionary):

> ['also', 'football', 'games', 'john', 'likes', 'mary', 'movies', 'to', 'too', 'watch']

该字典中包含10个单词, 每个单词有唯一索引, ***注意它们的顺序和出现在句子中的顺序没有关联. 根据这个字典,*** 我们能将上述两句话重新表达为下述两个向量:

> [0 0 0 1 2 1 2 1 1 1]
>  [1 1 1 1 1 0 0 1 0 1]

这两个向量共包含10个元素, 其中第i个元素表示字典中第i个单词在句子中出现的次数. 因此BoW模型可认为是一种 **统计直方图** (histogram)*. 在文本检索和处理应用中, 可以通过该模型很方便的计算***词频***.

但是从上面我们也能够看出，在构造文档向量的过程中可以看到，我们并没有表达单词在原来句子中出现的次序.

### 适用场景

现在想象在一个巨大的文档集合D，里面一共有M个文档，而文档里面的所有单词提取出来后，一起构成一个包含N个单词的词典，利用Bag-of-words模型，每个文档都可以被表示成为一个N维向量。

变为N维向量之后，很多问题就变得非常好解了，计算机非常擅长于处理数值向量，我们可以通过余弦来求两个文档之间的相似度，也可以将这个向量作为特征向量送入分类器进行主题分类等一系列功能中去。

### Example with code

scikit-learn中的 CountVectorizer() 函数实现了BOW模型，下面来看看用法：

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    "John likes to watch movies, Mary likes movies too",
    "John also likes to watch football games",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())
```
