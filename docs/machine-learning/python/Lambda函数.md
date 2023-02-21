#  lambda 函数

Python中，lambda函数也叫匿名函数，及即没有具体名称的函数，它允许快速定义单行函数，类似于C语言的宏，可以用在任何需要函数的地方。这区别于def定义的函数。

## lambda 特性

- lambda 函数是匿名的：

所谓匿名函数，通俗地说就是没有名字的函数，lambda函数没有名字。

- lambda 函数有输入和输出：

输入是传入到参数列表argument_list的值，输出是根据表达式expression计算得到的值。

- lambda 函数拥有自己的命名空间：

不能访问自己参数列表之外或全局命名空间里的参数，只能完成非常简单的功能。
常见的lambda函数示例：

```python
lambda x, y: x*y			# 函数输入是x和y，输出是它们的积x*y
lambda:None					# 函数没有输入参数，输出是None
lambda *args: sum(args)		# 输入是任意个数参数，输出是它们的和(隐性要求输入参数必须能进行算术运算)
lambda **kwargs: 1			# 输入是任意键值对参数，输出是1
```

## lambda 与 def 的区别

- 1）def 创建的方法是有名称的，而lambda没有。
- 2）lambda会返回一个函数对象，但这个对象不会赋给一个标识符，而def则会把函数对象赋值给一个变量（函数名）。
- 3）lambda只是一个表达式，而def则是一个语句。
- 4）lambda表达式” : “后面，只能有一个表达式，def则可以有多个。
- 5）像 if 或 for 或 print 等语句不能用于 lambda 中，def 可以。
- 6）lambda一般用来定义简单的函数，而def可以定义复杂的函数。
- 6）lambda函数不能共享给别的程序调用，def可以。

## lambda 语法格式

lambda 变量 : 要执行的语句

```shell
lambda [arg1 [, agr2,.....argn]] : expression
```

其中，lambda 是 Python 预留的关键字，[arg…] 和 expression 由用户自定义。

具体介绍如下:
**[arg…] 是参数列表**，它的结构与 Python 中函数(function)的参数列表是一样的。
[arg…] 可以有非常多的形式。例如：

- 单个参数的：

```python
g = lambda x : x ** 2
print g(3)
9
```

- 多个参数的：

```python
g = lambda x, y, z : (x + y) ** z
print g(1,2,2)
9
```

**expression 是一个参数表达式**，表达式中出现的参数需要在`[arg......]`中有定义，并且表达式只能是单行的，只能有一个表达式。以下都是合法的表达式：

```python
None
a + b
sum(a)
1 if a >10 else 0
......
```

## lambda 常见用法

由于lambda语法是固定的，其本质上只有一种用法，那就是定义一个lambda函数。
    在实际中，根据这个lambda函数应用场景的不同，可以将lambda函数的用法扩展为以下几种：

1、将lambda函数赋值给一个变量，通过这个变量间接调用该lambda函数。
示例：

`add = lambda x, y: x+y
` 相当于定义了加法函数`lambda x, y: x+y`，并将其赋值给变量add，这样变量add就指向了具有加法功能的函数。
这时我们如果执行 `add(1, 2)`，其输出结果就为 3。

2、将lambda函数赋值给其他函数，从而将其他函数用该lambda函数替换。
示例：

```python
# 为了把标准库time中的函数sleep的功能屏蔽(Mock)，我们可以在程序初始化时调用：
time.sleep=lambda x: None
# 这样，在后续代码中调用time库的sleep函数将不会执行原有的功能。
# 例如：
time.sleep(3)	# 程序不会休眠 3 秒钟，而是因为lambda输出为None，所以这里结果是什么都不做
```

**3、将lambda函数作为参数传递给其他函数。**

典型的用法就是下面我们常见的几种高阶函数。

## lambda 用法之高阶函数

### map() 函数：

> 描述：
> map() 会根据提供的函数对指定序列做映射。
>
> 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
>
> 语法：
> map(function, iterable, ...)
>
> 参数：
> function ----> 函数
>
> iterable ----> 一个或多个序列

实例：

```python
# ===========一般写法：===========
# 1、计算平方数
def square(x):
	return x ** 2

map(square, [1,2,3,4,5])	# 计算列表各个元素的平方
# 结果：
[1, 4, 9, 16, 25]

# ===========匿名函数写法：============
# 2、计算平方数，lambda 写法
map(lambda x: x ** 2, [1, 2, 3, 4, 5])
# 结果：
[1, 4, 9, 16, 25]	 

# 3、提供两个列表，将其相同索引位置的列表元素进行相加
map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
# 结果：
[3, 7, 11, 15, 19]
```

### reduce() 函数：

> 描述：
>
> reduce() 函数会对参数序列中元素进行累积。
>
> 函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。
>
> 语法：
>
> reduce(function, iterable[, initializer])
>
> 参数：
>
> function  ----> 函数，有两个参数
>
> iterable   ----> 可迭代对象
>
> initializer ----> 可选，初始参数
>
> 返回值：
>
> 返回函数计算结果。

**实例：**

```python
# ===========一般写法：===========
# 1、两数相加
def add(x, y):            
	return x + y

reduce(add, [1, 3, 5, 7, 9])    # 计算列表元素和：1+3+5+7+9
# 结果：
25

"""
===========执行步骤解析：===========
调用 reduce(add, [1, 3, 5, 7, 9])时，reduce函数将做如下计算：
1	先计算头两个元素：f(1, 3)，结果为4；
2	再把结果和第3个元素计算：f(4, 5)，结果为9；
3	再把结果和第4个元素计算：f(9, 7)，结果为16；
4	再把结果和第5个元素计算：f(16, 9)，结果为25；
5	由于没有更多的元素了，计算结束，返回结果25。
"""
# ===========匿名函数写法：===========
# 2、两数相加，lambda 写法
reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
# 结果：
15

# 当然求和运算可以直接用Python内建函数sum()，没必要动用reduce。
	
# 3、但是如果要把序列 [1, 3, 5, 7, 9] 变换成整数 13579，reduce就可以派上用场：
from functools import reduce

def fn(x, y):
	return x * 10 + y

reduce(fn, [1, 3, 5, 7, 9])
# 结果：
13579
```

### sorted() 函数：

> 描述：
> sorted() 函数对所有可迭代的对象进行排序操作。
>
> sort 与 sorted 区别：
> sort 是 list 的一个方法，而 sorted 可以对所有可迭代的对象进行排序操作。
> list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
>
> 语法：
>
> sorted(iterable[, cmp[, key[, reverse]]])
>
> 参数说明：
>
> iterable  ----> 可迭代对象。
>
> cmp       ----> 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
>
> key        ----> 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
>
> reverse  ----> 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
>
> 返回值：
>
> 返回重新排序的列表。

**示例：**

```python
# ===========一般用法：===========
# 1、简单排序
a = [5,7,6,3,4,1,2]
b = sorted(a)       # 使用sorted，保留原列表，不改变列表a的值
print(a)
[5, 7, 6, 3, 4, 1, 2]
print(b)
[1, 2, 3, 4, 5, 6, 7]

# ===========匿名函数用法：===========
L=[('b',2),('a',1),('c',3),('d',4)]
# 2、利用参数 cmp 排序
sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))
# 结果：
[('a', 1), ('b', 2), ('c', 3), ('d', 4)]
# 3、利用参数 key 排序
sorted(L, key=lambda x:x[1])
# 结果：
[('a', 1), ('b', 2), ('c', 3), ('d', 4)]

# 4、按年龄升序
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=lambda s: s[2])
# 结果：
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
# 5、按年龄降序
sorted(students, key=lambda s: s[2], reverse=True)
# 结果：
[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
```

### filter() 函数：

> 描述：
> filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
>
> 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
>
> 语法：
>
> filter(function, iterable)
>
> 参数：
>
> function ----> 判断函数。
>
> iterable  ----> 可迭代对象。
>
> 返回值：
>
> Pyhton2.7 返回列表，Python3.x 返回迭代器对象，具体内容可以查看：Python3 filter() 函数

**实例：**

```python
# ===========一般用法：===========
# 1、过滤出列表中的所有奇数
def is_odd(n):
	return n % 2 == 1
		 
newlist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(list(newlist))
# 结果： [1, 3, 5, 7, 9]

# ===========匿名函数用法：===========
# 2、将列表[1, 2, 3]中能够被3整除的元素过滤出来
newlist = filter(lambda x: x % 3 == 0, [1, 2, 3])
print(list(newlist))
# 结果： [3]
```