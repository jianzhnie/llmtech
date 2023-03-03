

# 计算机专业学习路线

## 课程学习进度

|       课程       |                   课程网站                    |                         B站                         | 难度系数 | 进度 |
| :--------------: | :-------------------------------------------: | :-------------------------------------------------: | :------: | ---- |
| 计算机科学速成课 |                       -                       | [学习](https://www.bilibili.com/video/BV1EW411u7th) |   入门   |      |
|  数据结构与算法  | [The Algorithms](https://the-algorithms.com/) |                                                     |   中等   |      |
|     算法导论     |                                               |                                                     |   中等   |      |
|    计算机系统    |                                               |                                                     |    难    |      |
|     操作系统     |                                               |                                                     |    难    |      |
|    计算机网络    |                                               |                                                     |    难    |      |
|      编译器      |                                               |                                                     |    难    |      |
|    分布式计算    |                                               |                                                     |   深入   |      |

## 自学计算机科学

如果你是一个自学成才的工程师，或者从编程培训班毕业，那么你很有必要学习计算机科学。幸运的是，不必为此花上数年光阴和不菲费用去攻读一个学位：仅仅依靠自己，你就可以获得世界一流水平的教育💸。

互联网上，到处都有许多的学习资源，然而精华与糟粕并存。你所需要的，不是一个诸如「200+ 免费在线课程」的清单，而是以下问题的答案：

* 你应当学习 **哪些科目**，为什么？
* 对于这些科目，**最好的书籍或者视频课程** 是什么？

在这份指引中，我们尝试对这些问题做出确定的回答。

大致按照列出的顺序，借助我们所建议的教材或者视频课程（但是最好二者兼用），学习如下的九门科目。目标是先花 100 到 200 个小时学习完每一个科目，然后在你职业生涯中，不时温习其中的精髓🚀。

| 科目                                  | 为何要学？                                                   | 最佳书籍                                                     | 最佳视频                          |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------- |
| [编程](#编程)                         | 不要做一个「永远没彻底搞懂」诸如递归等概念的程序员。         | [《计算机程序的构造和解释》](https://book.douban.com/subject/1148282/) | Brian Harvey’s Berkeley CS 61A    |
| [计算机系统结构](#计算机系统结构)     | 如果你对于计算机如何工作没有具体的概念，那么你所做出的所有高级抽象都是空中楼阁。 | [《深入理解计算机系统》](https://book.douban.com/subject/26912767/) | Berkeley CS 61C                   |
| [算法与数据结构](#算法和数据结构)     | 如果你不懂得如何使用栈、队列、树、图等常见数据结构，遇到有难度的问题时，你将束手无策。 | [《算法设计手册》](https://book.douban.com/subject/4048566/) | Steven Skiena’s lectures          |
| [数学知识](#数学知识)                 | 计算机科学基本上是应用数学的一个「跑偏的」分支，因此学习数学将会给你带来竞争优势。 | [《计算机科学中的数学》](https://book.douban.com/subject/33396340/) | Tom Leighton’s MIT 6.042J         |
| [操作系统](#操作系统)                 | 你所写的代码，基本上都由操作系统来运行，因此你应当了解其运作的原理。 | [《操作系统导论》](https://book.douban.com/subject/33463930/) | Berkeley CS 162                   |
| [计算机网络](#计算机网络)             | 互联网已然势不可挡：理解工作原理才能解锁全部潜力。           | [《计算机网络：自顶向下方法》](https://book.douban.com/subject/30280001/) | Stanford CS 144                   |
| [数据库](#数据库)                     | 对于多数重要程序，数据是其核心，然而很少人理解数据库系统的工作原理。 | *[Readings in Database Systems](https://book.douban.com/subject/2256069/)* （暂无中译本） | Joe Hellerstein’s Berkeley CS 186 |
| [编程语言与编译器](#编程语言与编译器) | 若你懂得编程语言和编译器如何工作，你就能写出更好的代码，更轻松地学习新的编程语言。 | *[Crafting Interpreters](https://craftinginterpreters.com/)* | Alex Aiken’s course on Lagunita   |
| [分布式系统](#分布式系统)             | 如今，**多数** 系统都是分布式的。                            | [《数据密集型应用系统设计》](https://book.douban.com/subject/30329536/) | MIT 6.824                         |

### 计算机系统

参考课程：华盛顿大学[CSE351: The Hardware/Software Interface](http://courses.cs.washington.edu/courses/cse351/)

参考书籍：[深入理解计算机系统(CSAPP)](http://product.dangdang.com/24106647.html)

参考视频：B站 [Washington CSE351 2017](https://www.bilibili.com/video/BV1Zt411s7Gg)

产出目标：完成[CSAPP书籍配套的所有Labs](http://csapp.cs.cmu.edu/3e/labs.html)

挑战难度：4星

这门课程是系统编程基础，也是后续操作系统/网络/数据库/编译等课程的基础，相关内容是通向系统架构师的基本功。这门课比较贴近企业实战，对动手能力要求很高，课程一大目标是要程序员写出对机器友好的高性能代码。

### 数据结构

参考课程：伯克利大学[CS61B Data Structures](https://sp19.datastructur.es/)

参考书籍：Head First Java + 数据结构书自选

参考视频：B站 [UCB CS 61B Data Structures](https://www.bilibili.com/video/BV1EJ411n72e)

产出目标：完成CS 61B站点上的所有Labs/Homeworks/Projects。

挑战难度：4星

说明：数据结构的重要性毋庸置疑，伯克利的CS课程都是比较偏向实战型工程师的，纯理论的东西相对少。本课的重点是树立抽象编程思维，务必把所有Labs/Homeworks/Projects都搞定。

### 操作系统

参考课程：麻省理工MIT 6.828 [Operating System Engineering](https://pdos.csail.mit.edu/6.828/2018/index.html)

参考书籍：[操作系统导论(Operating Systems: Three Easy Pieces)](http://product.dangdang.com/27882546.html)

参考视频：B站 [HMC CS 134 2019 Operating System](https://www.bilibili.com/video/av47977122)

产出目标：完成MIT 6.828站点上的所有7个Labs

挑战难度：5星

说明：6.828是MIT的神课，这门课难度不小，含金量也不小。如果能把所有实验都搞定，对操作系统的认识会有质的飞跃。

### 计算机网络

参考课程：斯坦福 [CS 144 Introduction to Computer Networking](https://cs144.github.io/)

参考书籍：[计算机网络：自顶向下方法](http://product.dangdang.com/25299722.html)

参考视频：B站 [斯坦福大学：CS144 计算机网络介绍](https://www.bilibili.com/video/BV137411Z7LR)

产出目标：完成CS 144 站点上的所有8个Labs。

挑战难度：4星

说明：计算机网络知识和技能，是互联网应用开发的基础，也是成为系统架构师的基础。这门CS 144和配套书《计算机网络：自顶向下方法》，是目前最佳的学习计算机网络基础的课程和参考书。这也是一门投入产出比比较

### 编译原理

参考书籍：[Crafting Interpreters](https://www.craftinginterpreters.com/contents.html) 或者 [Write an Interpreter in Go](https://www.amazon.com/Writing-Interpreter-Go-Thorsten-Ball/dp/3982016118)

参考视频：B站 [CS143 斯坦福编译原理](https://www.bilibili.com/video/BV1cE411f78c)

产出目标：参考[Crafting Interpreters](https://github.com/munificent/craftinginterpreters)，使用Java或者golang语言(或其它你熟悉的语言)，实现Lox小型编程语言。

或者，参考[Write an Interpreter in Go](https://interpreterbook.com/)，或[Write A Compiler in Go](https://compilerbook.com/)，使用Java语言实现Monkey小型语言。

挑战难度：5星

说明：视频可以不看，但是一定要自己动手实现一个小语言解释器或者编译器。

### 数据库系统

参考课程：卡耐基梅隆CMU [15-445/645 Database Systems](https://15445.courses.cs.cmu.edu/fall2020/)

参考书籍：[数据库系统概念](http://product.dangdang.com/22632572.html)

参考视频：B站 [卡耐基梅隆大学15-445 数据库系统介绍](https://www.bilibili.com/video/BV1Cp4y1C7dv)

产出目标：参考[vanilladb项目](https://github.com/vanilladb/vanillacore)，使用golang语言实现clone版的vanilladb（原项目是Java实现的）。

挑战难度：5星

说明：视频/课程/书可以不看，但是一定要自己动手实现一个小型的数据库系统，包括服务器端的存储引擎、SQL解析器、查询引擎和JDBC访问接口。企业开发大部分是基于数据库的应用，如果要成为企业级架构师，必须对数据库底层实现有一定掌握。课程项目要求用golang，对golang语言不熟悉的，自己找资料自学，如果你按照课程计划坚持学到这门课，那么你已经具有足够基础，可以轻松pick up任何一门编程语言。

## 参考网站

- https://teachyourselfcs.com/
- https://github.com/izackwu/TeachYourselfCS-CN

- https://hackway.org/docs/cs/intro
- https://github.com/spring2go/cs_study_plan
