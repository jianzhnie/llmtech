

## TF-IDF

TF-IDF(term frequency–inverse document frequency)是一种用于信息检索与数据挖掘的常用加权技术，常用于挖掘文章中的关键词，而且算法简单高效，常被工业用于最开始的文本数据清洗。

TF-IDF有两层意思，一层是"词频"（Term Frequency，缩写为TF），另一层是"逆文档频率"（Inverse Document Frequency，缩写为IDF）。

假设我们现在有一片长文叫做《量化系统架构设计》词频高在文章中往往是停用词，“的”，“是”，“了”等，这些在文档中最常见但对结果毫无帮助、需要过滤掉的词，用TF可以统计到这些停用词并把它们过滤。当高频词过滤后就只需考虑剩下的有实际意义的词。

但这样又会遇到了另一个问题，我们可能发现"量化"、"系统"、"架构"这三个词的出现次数一样多。这是不是意味着，作为关键词，它们的重要性是一样的？事实上系统应该在其他文章比较常见，所以在关键词排序上，“量化”和“架构”应该排在“系统”前面，这个时候就需要IDF，IDF会给常见的词较小的权重，它的大小与一个词的常见程度成反比。

**当有TF(词频)和IDF(逆文档频率)后，将这两个词相乘，就能得到一个词的TF-IDF的值。某个词在文章中的TF-IDF越大，那么一般而言这个词在这篇文章的重要性会越高，所以通过计算文章中各个词的TF-IDF，由大到小排序，排在最前面的几个词，就是该文章的关键词。**

![img](https://pic3.zhimg.com/80/v2-e3748e7dd4056e0a98d7c9892d11cc52_1440w.jpg)

### **TF-IDF算法步骤**

第一步，计算词频：

![img](https://pic4.zhimg.com/80/v2-281a550de928afe343c055d06371cf77_1440w.jpg)

考虑到文章有长短之分，为了便于不同文章的比较，进行"词频"标准化。

![img](https://pic2.zhimg.com/80/v2-393435b342546a2f1736d1d755adb1cd_1440w.jpg)

第二步，计算逆文档频率：

这时，需要一个语料库（corpus），用来模拟语言的使用环境。

![img](https://pic2.zhimg.com/80/v2-1d5c436e04f497544d72fec6909a3fad_1440w.jpg)

如果一个词越常见，那么分母就越大，逆文档频率就越小越接近0。分母之所以要加1，是为了避免分母为0（即所有文档都不包含该词）。log表示对得到的值取对数。

第三步，计算TF-IDF：

![img](https://pic3.zhimg.com/80/v2-5560a4b2efa3330021b8b2ef13a471fe_1440w.jpg)

可以看到，TF-IDF与一个词在文档中的出现次数成正比，与该词在整个语言中的出现次数成反比。所以，自动提取关键词的算法就很清楚了，就是**计算出文档的每个词的TF-IDF值，然后按降序排列，取排在最前面的几个词。**

### **优缺点**

BOW模型有很多缺点，首先它没有考虑单词之间的顺序，其次它无法反应出一个句子的关键词，比如下面这个句子：

"John likes to play football, Mary likes too"

这个句子若用BOW模型，它的词表为：[‘football’, ‘john’, ‘likes’, ‘mary’, ‘play’, ‘to’, ‘too’]，则词向量表示为：[1 1 2 1 1 1 1]。若根据BOW模型提取这个句子的关键词，则为 “like”，但是显然这个句子的关键词应该为 “football”。而TF-IDF则可以解决这个问题。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X)
print(X.toarray())
```

TF-IDF的优点是简单快速，而且容易理解。缺点是有时候用**词频**来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。如果要体现词的上下文结构，那么你可能需要使用word2vec算法来支持。
