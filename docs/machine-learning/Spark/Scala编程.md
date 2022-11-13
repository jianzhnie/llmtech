# 导言: Scala 之旅



## Scala是什么？

Scala是一门现代的多范式语言，志在以简洁、优雅及类型安全的方式来表达常用的编程模型。它平滑地集成了面向对象和函数式语言的特性。

## Scala是面向对象的

鉴于[一切值都是对象](https://docs.scala-lang.org/zh-cn/tour/unified-types.html)，可以说Scala是一门纯面向对象的语言。对象的类型和行为是由[类](https://docs.scala-lang.org/zh-cn/tour/classes.html)和[特质](https://docs.scala-lang.org/zh-cn/tour/traits.html)来描述的。类可以由子类化和一种灵活的、基于mixin的组合机制（它可作为多重继承的简单替代方案）来扩展。

## Scala是函数式的

鉴于[一切函数都是值](https://docs.scala-lang.org/zh-cn/tour/unified-types.html)，又可以说Scala是一门函数式语言。Scala为定义匿名函数提供了[轻量级的语法](https://docs.scala-lang.org/zh-cn/tour/basics.html#函数)，支持[高阶函数](https://docs.scala-lang.org/zh-cn/tour/higher-order-functions.html)，允许[函数嵌套](https://docs.scala-lang.org/zh-cn/tour/nested-functions.html)及[柯里化](https://docs.scala-lang.org/zh-cn/tour/multiple-parameter-lists.html)。Scala的[样例类](https://docs.scala-lang.org/zh-cn/tour/case-classes.html)和内置支持的[模式匹配](https://docs.scala-lang.org/zh-cn/tour/pattern-matching.html)代数模型在许多函数式编程语言中都被使用。对于那些并非类的成员函数，[单例对象](https://docs.scala-lang.org/zh-cn/tour/singleton-objects.html)提供了便捷的方式去组织它们。

此外，通过对提取器的一般扩展，Scala的模式匹配概念使用了[right-ignoring序列模式](https://docs.scala-lang.org/zh-cn/tour/regular-expression-patterns.html)，自然地延伸到[XML数据的处理](https://github.com/scala/scala-xml/wiki/XML-Processing)。其中，[for表达式](https://docs.scala-lang.org/zh-cn/tour/for-comprehensions.html)对于构建查询很有用。这些特性使得Scala成为开发web服务等程序的理想选择。

## Scala是静态类型的

Scala配备了一个拥有强大表达能力的类型系统，它可以静态地强制以安全、一致的方式使用抽象。典型来说，这个类型系统支持：

- [泛型类](https://docs.scala-lang.org/zh-cn/tour/generic-classes.html)
- [型变注解](https://docs.scala-lang.org/zh-cn/tour/variances.html)
- [上](https://docs.scala-lang.org/zh-cn/tour/upper-type-bounds.html)、[下](https://docs.scala-lang.org/zh-cn/tour/lower-type-bounds.html) 类型边界
- 作为对象成员的[内部类](https://docs.scala-lang.org/zh-cn/tour/inner-classes.html)和[抽象类型](https://docs.scala-lang.org/zh-cn/tour/abstract-type-members.html)
- [复合类型](https://docs.scala-lang.org/zh-cn/tour/compound-types.html)
- [显式类型的自我引用](https://docs.scala-lang.org/zh-cn/tour/self-types.html)
- [隐式参数](https://docs.scala-lang.org/zh-cn/tour/implicit-parameters.html)和[隐式转化](https://docs.scala-lang.org/zh-cn/tour/implicit-conversions.html)
- [多态方法](https://docs.scala-lang.org/zh-cn/tour/polymorphic-methods.html)

[类型推断](https://docs.scala-lang.org/zh-cn/tour/type-inference.html)让用户不需要标明额外的类型信息。这些特性结合起来为安全可重用的编程抽象以及类型安全的扩展提供了强大的基础。

## Scala是可扩展的

在实践中，特定领域应用的发展往往需要特定领域的语言扩展。Scala提供了一种语言机制的独特组合方式，使得可以方便地以库的形式添加新的语言结构。

很多场景下，这些扩展可以不通过类似宏（macros）的元编程工具完成。例如：

- [隐式类](https://docs.scala-lang.org/overviews/core/implicit-classes.html)允许给已有的类型添加扩展方法。
- [字符串插值](https://docs.scala-lang.org/overviews/core/string-interpolation.html)可以让用户使用自定义的插值器进行扩展。

## Scala的互操作性

Scala设计的目标是与流行的Java运行环境（JRE）进行良好的互操作，特别是与主流的面向对象编程语言——Java的互操作尽可能的平滑。Java的最新特性如函数接口（SAMs）、[lambda表达式](https://docs.scala-lang.org/zh-cn/tour/higher-order-functions.html)、[注解](https://docs.scala-lang.org/zh-cn/tour/annotations.html)及[泛型类](https://docs.scala-lang.org/zh-cn/tour/generic-classes.html) 在Scala中都有类似的实现。

另外有些Java中并没有的特性，如[缺省参数值](https://docs.scala-lang.org/zh-cn/tour/default-parameter-values.html)和[带名字的参数](https://docs.scala-lang.org/zh-cn/tour/named-arguments.html)等，也是尽可能地向Java靠拢。Scala拥有类似Java的编译模型（独立编译、动态类加载），且允许使用已有的成千上万的高质量类库。



# Scala 基础



## 表达式

表达式是可计算的语句。

```
1 + 1
```

你可以使用`println`来输出表达式的结果。

```
println(1) // 1
println(1 + 1) // 2
println("Hello!") // Hello!
println("Hello," + " world!") // Hello, world!
```

### 常量（`Values`）

你可以使用`val`关键字来给表达式的结果命名。

```
val x = 1 + 1
println(x) // 2
```

对于结果比如这里的`x`的命名，被称为常量（`values`）。引用一个常量（`value`）不会再次计算。

常量（`values`）不能重新被赋值。

```
x = 3 // This does not compile.
```

常量（`values`）的类型可以被推断，或者你也可以显示地声明类型，例如：

```
val x: Int = 1 + 1
```

注意下，在标识符`x`的后面、类型声明`Int`的前面，还需要一个冒号`:`。

### 变量

除了可以重新赋值，变量和常量类似。你可以使用`var`关键字来定义一个变量。

```
var x = 1 + 1
x = 3 // This compiles because "x" is declared with the "var" keyword.
println(x * x) // 9
```

和常量一样，你可以显示地声明类型：

```
var x: Int = 1 + 1
```

## 代码块（Blocks）

你可以组合几个表达式，并且用`{}`包围起来。我们称之为代码块（block）。

代码块中最后一个表达式的结果，也正是整个块的结果。

```
println({
  val x = 1 + 1
  x + 1
}) // 3
```

## 函数

函数是带有参数的表达式。

你可以定义一个匿名函数（即没有名字），来返回一个给定整数加一的结果。

```
(x: Int) => x + 1
```

`=>`的左边是参数列表，右边是一个包含参数的表达式。

你也可以给函数命名。

```
val addOne = (x: Int) => x + 1
println(addOne(1)) // 2
```

函数可带有多个参数。

```
val add = (x: Int, y: Int) => x + y
println(add(1, 2)) // 3
```

或者不带参数。

```
val getTheAnswer = () => 42
println(getTheAnswer()) // 42
```

## 方法

方法的表现和行为和函数非常类似，但是它们之间有一些关键的差别。

方法由`def`关键字定义。`def`后面跟着一个名字、参数列表、返回类型和方法体。

```
def add(x: Int, y: Int): Int = x + y
println(add(1, 2)) // 3
```

注意返回类型是怎么在函数列表和一个冒号`: Int`之后声明的。

方法可以接受多个参数列表。

```
def addThenMultiply(x: Int, y: Int)(multiplier: Int): Int = (x + y) * multiplier
println(addThenMultiply(1, 2)(3)) // 9
```

或者没有参数列表。

```
def name: String = System.getProperty("user.name")
println("Hello, " + name + "!")
```

还有一些其他的区别，但是现在你可以认为方法就是类似于函数的东西。

方法也可以有多行的表达式。

```
def getSquareString(input: Double): String = {
  val square = input * input
  square.toString
}
println(getSquareString(2.5)) // 6.25
```

方法体的最后一个表达式就是方法的返回值。（Scala中也有一个`return`关键字，但是很少使用）

## 类

你可以使用`class`关键字定义一个类，后面跟着它的名字和构造参数。

```
class Greeter(prefix: String, suffix: String) {
  def greet(name: String): Unit =
    println(prefix + name + suffix)
}
```

`greet`方法的返回类型是`Unit`，表明没有什么有意义的需要返回。它有点像Java和C语言中的`void`。（不同点在于每个Scala表达式都必须有值，事实上有个`Unit`类型的单例值，写作`()`，它不携带任何信息）

你可以使用`new`关键字创建一个类的实例。

```
val greeter = new Greeter("Hello, ", "!")
greeter.greet("Scala developer") // Hello, Scala developer!
```

我们将在[后面](https://docs.scala-lang.org/zh-cn/tour/classes.html)深入介绍类。

## 样例类

Scala有一种特殊的类叫做样例类（case class）。默认情况下，样例类一般用于不可变对象，并且可作值比较。你可以使用`case class`关键字来定义样例类。

```
case class Point(x: Int, y: Int)
```

你可以不用`new`关键字来实例化样例类。

```
val point = Point(1, 2)
val anotherPoint = Point(1, 2)
val yetAnotherPoint = Point(2, 2)
```

并且它们的值可以进行比较。

```
if (point == anotherPoint) {
  println(point + " and " + anotherPoint + " are the same.")
} else {
  println(point + " and " + anotherPoint + " are different.")
} // Point(1,2) and Point(1,2) are the same.

if (point == yetAnotherPoint) {
  println(point + " and " + yetAnotherPoint + " are the same.")
} else {
  println(point + " and " + yetAnotherPoint + " are different.")
} // Point(1,2) and Point(2,2) are different.
```

关于样例类，还有不少内容我们乐于介绍，并且我们确信你会爱上它们。我们会在[后面](https://docs.scala-lang.org/zh-cn/tour/case-classes.html)深入介绍它们。

## 对象

对象是它们自己定义的单实例，你可以把它看作它自己的类的单例。

你可以使用`object`关键字定义对象。

```
object IdFactory {
  private var counter = 0
  def create(): Int = {
    counter += 1
    counter
  }
}
```

你可以通过引用它的名字来访问一个对象。

```
val newId: Int = IdFactory.create()
println(newId) // 1
val newerId: Int = IdFactory.create()
println(newerId) // 2
```

我们会在[后面](https://docs.scala-lang.org/zh-cn/tour/singleton-objects.html)深入介绍它们。

## 特质

特质是包含某些字段和方法的类型。可以组合多个特质。

你可以使用`trait`关键字定义特质。

```
trait Greeter {
  def greet(name: String): Unit
}
```

特质也可以有默认的实现。

```
trait Greeter {
  def greet(name: String): Unit =
    println("Hello, " + name + "!")
}
```

你可以使用`extends`关键字来继承特质，使用`override`关键字来覆盖默认的实现。

```
class DefaultGreeter extends Greeter

class CustomizableGreeter(prefix: String, postfix: String) extends Greeter {
  override def greet(name: String): Unit = {
    println(prefix + name + postfix)
  }
}

val greeter = new DefaultGreeter()
greeter.greet("Scala developer") // Hello, Scala developer!

val customGreeter = new CustomizableGreeter("How are you, ", "?")
customGreeter.greet("Scala developer") // How are you, Scala developer?
```

这里，`DefaultGreeter`仅仅继承了一个特质，它还可以继承多个特质。

我们会在[后面](https://docs.scala-lang.org/zh-cn/tour/traits.html)深入介绍特质。

## 主方法

主方法是一个程序的入口点。JVM要求一个名为`main`的主方法，接受一个字符串数组的参数。

通过使用对象，你可以如下所示来定义一个主方法。

```
object Main {
  def main(args: Array[String]): Unit =
    println("Hello, Scala developer!")
}
```



# Scala 类



Scala中的类是用于创建对象的蓝图，其中包含了方法、常量、变量、类型、对象、特质、类，这些统称为成员。类型、对象和特质将在后面的文章中介绍。

## 类定义

一个最简的类的定义就是关键字`class`+标识符，类名首字母应大写。

```
class User

val user1 = new User
```

关键字`new`被用于创建类的实例。`User`由于没有定义任何构造器，因而只有一个不带任何参数的默认构造器。然而，你通常需要一个构造器和类体。下面是类定义的一个例子：

```scala
class Point(var x: Int, var y: Int) {

  def move(dx: Int, dy: Int): Unit = {
    x = x + dx
    y = y + dy
  }

  override def toString: String =
    s"($x, $y)"
}

val point1 = new Point(2, 3)
point1.x  // 2
println(point1)  // prints (2, 3)
```

`Point`类有4个成员：变量`x`和`y`，方法`move`和`toString`。与许多其他语言不同，主构造方法在类的签名中`(var x: Int, var y: Int)`。`move`方法带有2个参数，返回无任何意义的`Unit`类型值`()`。这一点与Java这类语言中的`void`相当。另外，`toString`方法不带任何参数但是返回一个`String`值。因为`toString`覆盖了[`AnyRef`](https://docs.scala-lang.org/zh-cn/tour/unified-types.html)中的`toString`方法，所以用了`override`关键字标记。

## 构造器

构造器可以通过提供一个默认值来拥有可选参数：

```scala
class Point(var x: Int = 0, var y: Int = 0)

val origin = new Point  // x and y are both set to 0
val point1 = new Point(1)
println(point1.x)  // prints 1
```

在这个版本的`Point`类中，`x`和`y`拥有默认值`0`所以没有必传参数。然而，因为构造器是从左往右读取参数，所以如果仅仅要传个`y`的值，你需要带名传参。

```scala
class Point(var x: Int = 0, var y: Int = 0)
val point2 = new Point(y=2)
println(point2.y)  // prints 2
```

这样的做法在实践中有利于使得表达明确无误。

## 私有成员和Getter/Setter语法

成员默认是公有（`public`）的。使用`private`访问修饰符可以在类外部隐藏它们。

```scala
class Point {
  private var _x = 0
  private var _y = 0
  private val bound = 100

  def x = _x
  def x_= (newValue: Int): Unit = {
    if (newValue < bound) _x = newValue else printWarning
  }

  def y = _y
  def y_= (newValue: Int): Unit = {
    if (newValue < bound) _y = newValue else printWarning
  }

  private def printWarning = println("WARNING: Out of bounds")
}

val point1 = new Point
point1.x = 99
point1.y = 101 // prints the warning
```

在这个版本的`Point`类中，数据存在私有变量`_x`和`_y`中。`def x`和`def y`方法用于访问私有数据。`def x_=`和`def y_=`是为了验证和给`_x`和`_y`赋值。注意下对于setter方法的特殊语法：这个方法在getter方法的后面加上`_=`，后面跟着参数。

主构造方法中带有`val`和`var`的参数是公有的。然而由于`val`是不可变的，所以不能像下面这样去使用。

```scala
class Point(val x: Int, val y: Int)
val point = new Point(1, 2)
point.x = 3  // <-- does not compile
```

不带`val`或`var`的参数是私有的，仅在类中可见。

```scala
class Point(x: Int, y: Int)
val point = new Point(1, 2)
point.x  // <-- does not compile
```



## 默认参数值

Scala具备给参数提供默认值的能力，这样调用者就可以忽略这些具有默认值的参数。

```scala
def log(message: String, level: String = "INFO") = println(s"$level: $message")

log("System starting")  // prints INFO: System starting
log("User not found", "WARNING")  // prints WARNING: User not found
```

上面的参数level有默认值，所以是可选的。最后一行中传入的参数`"WARNING"`重写了默认值`"INFO"`。在Java中，我们可以通过带有可选参数的重载方法达到同样的效果。不过，只要调用方忽略了一个参数，其他参数就必须要带名传入。

```scala
class Point(val x: Double = 0, val y: Double = 0)

val point1 = new Point(y = 1)
```

这里必须带名传入`y = 1`。

注意从Java代码中调用时，Scala中的默认参数则是必填的（非可选），如：

```scala
// Point.scala
class Point(val x: Double = 0, val y: Double = 0)
// Main.java
public class Main {
    public static void main(String[] args) {
        Point point = new Point(1);  // does not compile
    }
}
```



## 命名参数



当调用方法时，实际参数可以通过其对应的形式参数的名称来标记：

```scala
def printName(first: String, last: String): Unit = {
  println(first + " " + last)
}

printName("John", "Smith")  // Prints "John Smith"
printName(first = "John", last = "Smith")  // Prints "John Smith"
printName(last = "Smith", first = "John")  // Prints "John Smith"
```

注意使用命名参数时，顺序是可以重新排列的。 但是，如果某些参数被命名了，而其他参数没有，则未命名的参数要按照其方法签名中的参数顺序放在前面。

```scala
printName(last = "Smith", "john") // error: positional after named argument
```

注意调用 Java 方法时不能使用命名参数。



# Scala 特质


特质 (Traits) 用于在类 (Class)之间共享程序接口 (Interface)和字段 (Fields)。 它们类似于Java 8的接口。 类和对象 (Objects)可以扩展特质，但是特质不能被实例化，因此特质没有参数。

## 定义一个特质

最简化的特质就是关键字trait+标识符：

```scala
trait HairColor
```

特征作为泛型类型和抽象方法非常有用。

```scala
trait Iterator[A] {
  def hasNext: Boolean
  def next(): A
}
```

扩展 `trait Iterator [A]` 需要一个类型 `A` 和实现方法`hasNext`和`next`。

## 使用特质

使用 `extends` 关键字来扩展特征。然后使用 `override` 关键字来实现trait里面的任何抽象成员：

```scala
trait Iterator[A] {
  def hasNext: Boolean
  def next(): A
}

class IntIterator(to: Int) extends Iterator[Int] {
  private var current = 0
  override def hasNext: Boolean = current < to
  override def next(): Int =  {
    if (hasNext) {
      val t = current
      current += 1
      t
    } else 0
  }
}


val iterator = new IntIterator(10)
iterator.next()  // returns 0
iterator.next()  // returns 1
```

这个类 `IntIterator` 将参数 `to` 作为上限。它扩展了 `Iterator [Int]`，这意味着方法 `next` 必须返回一个Int。

## 子类型

凡是需要特质的地方，都可以由该特质的子类型来替换。

```scala
import scala.collection.mutable.ArrayBuffer

trait Pet {
  val name: String
}

class Cat(val name: String) extends Pet
class Dog(val name: String) extends Pet

val dog = new Dog("Harry")
val cat = new Cat("Sally")

val animals = ArrayBuffer.empty[Pet]
animals.append(dog)
animals.append(cat)
animals.foreach(pet => println(pet.name))  // Prints Harry Sally
```

在这里 `trait Pet` 有一个抽象字段 `name` ，`name` 由Cat和Dog的构造函数中实现。最后一行，我们能调用`pet.name`的前提是它必须在特质Pet的子类型中得到了实现。



# Scala 元组



在 Scala 中，元组是一个可以容纳不同类型元素的类。 元组是不可变的。

当我们需要从函数返回多个值时，元组会派上用场。

元组可以创建如下：

```
val ingredient = ("Sugar" , 25):Tuple2[String, Int]
```

这将创建一个包含一个 String 元素和一个 Int 元素的元组。

Scala 中的元组包含一系列类：Tuple2，Tuple3等，直到 Tuple22。 因此，当我们创建一个包含 n 个元素（n 位于 2 和 22 之间）的元组时，Scala 基本上就是从上述的一组类中实例化 一个相对应的类，使用组成元素的类型进行参数化。 上例中，`ingredient` 的类型为 `Tuple2[String, Int]`。

## 访问元素

使用下划线语法来访问元组中的元素。 ‘tuple._n’ 取出了第 n 个元素（假设有足够多元素）。

```
println(ingredient._1) // Sugar

println(ingredient._2) // 25
```

## 解构元组数据

Scala 元组也支持解构。

```
val (name, quantity) = ingredient

println(name) // Sugar

println(quantity) // 25
```

元组解构也可用于模式匹配。

```scala
val planetDistanceFromSun = List(("Mercury", 57.9), ("Venus", 108.2), ("Earth", 149.6 ), ("Mars", 227.9), ("Jupiter", 778.3))

planetDistanceFromSun.foreach{ tuple => {

  tuple match {

      case ("Mercury", distance) => println(s"Mercury is $distance millions km far from Sun")

      case p if(p._1 == "Venus") => println(s"Venus is ${p._2} millions km far from Sun")

      case p if(p._1 == "Earth") => println(s"Blue planet is ${p._2} millions km far from Sun")

      case _ => println("Too far....")

    }

  }

}
```

或者，在 ‘for’ 表达式中。

```scala
val numPairs = List((2, 5), (3, -7), (20, 56))

for ((a, b) <- numPairs) {

  println(a * b)

}
```

类型 `Unit` 的值 `()` 在概念上与类型 `Tuple0` 的值 `()` 相同。 `Tuple0` 只能有一个值，因为它没有元素。

用户有时可能在元组和 case 类之间难以选择。 通常，如果元素具有更多含义，则首选 case 类。



# 通过混入（MIXIN）来组合类



当某个特质被用于组合类时，被称为混入。

```scala
abstract class A {
  val message: String
}
class B extends A {
  val message = "I'm an instance of class B"
}
trait C extends A {
  def loudMessage = message.toUpperCase()
}
class D extends B with C

val d = new D
println(d.message)  // I'm an instance of class B
println(d.loudMessage)  // I'M AN INSTANCE OF CLASS B
```

类`D`有一个父类`B`和一个混入`C`。一个类只能有一个父类但是可以有多个混入（分别使用关键字`extends`和`with`）。混入和某个父类可能有相同的父类。

现在，让我们看一个更有趣的例子，其中使用了抽象类：

```scala
abstract class AbsIterator {
  type T
  def hasNext: Boolean
  def next(): T
}
```

该类中有一个抽象的类型`T`和标准的迭代器方法。

接下来，我们将实现一个具体的类（所有的抽象成员`T`、`hasNext`和`next`都会被实现）：

```scala
class StringIterator(s: String) extends AbsIterator {
  type T = Char
  private var i = 0
  def hasNext = i < s.length
  def next() = {
    val ch = s charAt i
    i += 1
    ch
  }
}
```

`StringIterator`带有一个`String`类型参数的构造器，可用于对字符串进行迭代。（例如查看一个字符串是否包含某个字符）：

现在我们创建一个特质，也继承于`AbsIterator`。

```scala
trait RichIterator extends AbsIterator {
  def foreach(f: T => Unit): Unit = while (hasNext) f(next())
}
```

该特质实现了`foreach`方法——只要还有元素可以迭代（`while (hasNext)`），就会一直对下个元素(`next()`) 调用传入的函数`f: T => Unit`。因为`RichIterator`是个特质，可以不必实现`AbsIterator`中的抽象成员。

下面我们要把`StringIterator`和`RichIterator` 中的功能组合成一个类。

```scala
object StringIteratorTest extends App {
  class RichStringIter extends StringIterator("Scala") with RichIterator
  val richStringIter = new RichStringIter
  richStringIter foreach println
}
```

新的类`RichStringIter`有一个父类`StringIterator`和一个混入`RichIterator`。如果是单一继承，我们将不会达到这样的灵活性。





# 高阶函数



高阶函数是指使用其他函数作为参数、或者返回一个函数作为结果的函数。在Scala中函数是“一等公民”，所以允许定义高阶函数。这里的术语可能有点让人困惑，我们约定，使用函数值作为参数，或者返回值为函数值的“函数”和“方法”，均称之为“高阶函数”。

最常见的一个例子是Scala集合类（collections）的高阶函数`map`

```scala
val salaries = Seq(20000, 70000, 40000)
val doubleSalary = (x: Int) => x * 2
val newSalaries = salaries.map(doubleSalary) // List(40000, 140000, 80000)
```

函数`doubleSalary`有一个整型参数`x`，返回`x * 2`。一般来说，在`=>`左边的元组是函数的参数列表，而右边表达式的值则为函数的返回值。在第3行，函数`doubleSalary`被应用在列表`salaries`中的每一个元素。

为了简化压缩代码，我们可以使用匿名函数，直接作为参数传递给`map`:

```scala
val salaries = Seq(20000, 70000, 40000)
val newSalaries = salaries.map(x => x * 2) // List(40000, 140000, 80000)
```

注意在上述示例中`x`没有被显式声明为Int类型，这是因为编译器能够根据map函数期望的类型推断出`x`的类型。对于上述代码，一种更惯用的写法为：

```scala
val salaries = Seq(20000, 70000, 40000)
val newSalaries = salaries.map(_ * 2)
```

既然Scala编译器已经知道了参数的类型（一个单独的Int），你可以只给出函数的右半部分，不过需要使用`_`代替参数名（在上一个例子中是`x`）

## 强制转换方法为函数

你同样可以传入一个对象方法作为高阶函数的参数，这是因为Scala编译器会将方法强制转换为一个函数。

```scala
case class WeeklyWeatherForecast(temperatures: Seq[Double]) {

  private def convertCtoF(temp: Double) = temp * 1.8 + 32

  def forecastInFahrenheit: Seq[Double] = temperatures.map(convertCtoF) // <-- passing the method convertCtoF
}
```

在这个例子中，方法`convertCtoF`被传入`forecastInFahrenheit`。这是可以的，因为编译器强制将方法`convertCtoF`转成了函数`x => convertCtoF(x)` （注: `x`是编译器生成的变量名，保证在其作用域是唯一的）。

## 接收函数作为参数的函数

使用高阶函数的一个原因是减少冗余的代码。比方说需要写几个方法以通过不同方式来提升员工工资，若不使用高阶函数，代码可能像这样：

```scala
object SalaryRaiser {

  def smallPromotion(salaries: List[Double]): List[Double] =
    salaries.map(salary => salary * 1.1)

  def greatPromotion(salaries: List[Double]): List[Double] =
    salaries.map(salary => salary * math.log(salary))

  def hugePromotion(salaries: List[Double]): List[Double] =
    salaries.map(salary => salary * salary)
}
```

注意这三个方法的差异仅仅是提升的比例不同，为了简化代码，其实可以把重复的代码提到一个高阶函数中：

```scala
object SalaryRaiser {

  private def promotion(salaries: List[Double], promotionFunction: Double => Double): List[Double] =
    salaries.map(promotionFunction)

  def smallPromotion(salaries: List[Double]): List[Double] =
    promotion(salaries, salary => salary * 1.1)

  def bigPromotion(salaries: List[Double]): List[Double] =
    promotion(salaries, salary => salary * math.log(salary))

  def hugePromotion(salaries: List[Double]): List[Double] =
    promotion(salaries, salary => salary * salary)
}
```

新的方法`promotion`有两个参数，薪资列表和一个类型为`Double => Double`的函数（参数和返回值类型均为Double），返回薪资提升的结果。

## 返回函数的函数

有一些情况你希望生成一个函数， 比如：

```scala
def urlBuilder(ssl: Boolean, domainName: String): (String, String) => String = {
  val schema = if (ssl) "https://" else "http://"
  (endpoint: String, query: String) => s"$schema$domainName/$endpoint?$query"
}

val domainName = "www.example.com"
def getURL = urlBuilder(ssl=true, domainName)
val endpoint = "users"
val query = "id=1"
val url = getURL(endpoint, query) // "https://www.example.com/users?id=1": String
```

注意urlBuilder的返回类型是`(String, String) => String`，这意味着返回的匿名函数有两个String参数，返回一个String。在这个例子中，返回的匿名函数是`(endpoint: String, query: String) => s"https://www.example.com/$endpoint?$query"`。



# 嵌套方法



在Scala中可以嵌套定义方法。例如以下对象提供了一个`factorial`方法来计算给定数值的阶乘：

```scala
 def factorial(x: Int): Int = {
    def fact(x: Int, accumulator: Int): Int = {
      if (x <= 1) accumulator
      else fact(x - 1, x * accumulator)
    }
    fact(x, 1)
 }

 println("Factorial of 2: " + factorial(2))
 println("Factorial of 3: " + factorial(3))
```

程序的输出为:

```
Factorial of 2: 2
Factorial of 3: 6
```



# 多参数列表（柯里化）



方法可以定义多个参数列表，当使用较少的参数列表调用多参数列表的方法时，会产生一个新的函数，该函数接收剩余的参数列表作为其参数。这被称为[柯里化](https://zh.wikipedia.org/wiki/柯里化)。

下面是一个例子，在Scala集合 `trait TraversableOnce` 定义了 `foldLeft`

```scala
def foldLeft[B](z: B)(op: (B, A) => B): B
```

`foldLeft`从左到右，以此将一个二元运算`op`应用到初始值`z`和该迭代器（traversable)的所有元素上。以下是该函数的一个用例：

从初值0开始, 这里 `foldLeft` 将函数 `(m, n) => m + n` 依次应用到列表中的每一个元素和之前累积的值上。

```scala
val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
val res = numbers.foldLeft(0)((m, n) => m + n)
print(res) // 55
```

多参数列表有更复杂的调用语法，因此应该谨慎使用，建议的使用场景包括：

## 单一的函数参数

在某些情况下存在单一的函数参数时，例如上述例子`foldLeft`中的`op`，多参数列表可以使得传递匿名函数作为参数的语法更为简洁。如果不使用多参数列表，代码可能像这样：

```scala
numbers.foldLeft(0, {(m: Int, n: Int) => m + n})
```

注意使用多参数列表时，我们还可以利用Scala的类型推断来让代码更加简洁（如下所示），而如果没有多参数列表，这是不可能的。

```scala
numbers.foldLeft(0)(_ + _)
```

像上述语句这样，我们可以给定多参数列表的一部分参数列表（如上述的`z`）来形成一个新的函数（partially applied function），达到复用的目的，如下所示：

```scala
val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
val numberFunc = numbers.foldLeft(List[Int]())_

val squares = numberFunc((xs, x) => xs:+ x*x)
print(squares.toString()) // List(1, 4, 9, 16, 25, 36, 49, 64, 81, 100)

val cubes = numberFunc((xs, x) => xs:+ x*x*x)
print(cubes.toString())  // List(1, 8, 27, 64, 125, 216, 343, 512, 729, 1000)
```

最后，`foldLeft` 和 `foldRight` 可以按以下任意一种形式使用，

```scala
val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

numbers.foldLeft(0)((sum, item) => sum + item) // Generic Form
numbers.foldRight(0)((sum, item) => sum + item) // Generic Form

numbers.foldLeft(0)(_+_) // Curried Form
numbers.foldRight(0)(_+_) // Curried Form
```

## 隐式（IMPLICIT）参数

如果要指定参数列表中的某些参数为隐式（implicit），应该使用多参数列表。例如：

```scala
def execute(arg: Int)(implicit ec: ExecutionContext) = ???
```





# 案例类（CASE CLASSES）



案例类（Case classes）和普通类差不多，只有几点关键差别，接下来的介绍将会涵盖这些差别。案例类非常适合用于不可变的数据。下一节将会介绍他们在[模式匹配](https://docs.scala-lang.org/zh-cn/tour/pattern-matching.html)中的应用。

## 定义一个案例类

一个最简单的案例类定义由关键字`case class`，类名，参数列表（可为空）组成：

```scala
case class Book(isbn: String)

val frankenstein = Book("978-0486282114")
```

注意在实例化案例类`Book`时，并没有使用关键字`new`，这是因为案例类有一个默认的`apply`方法来负责对象的创建。

当你创建包含参数的案例类时，这些参数是公开（public）的`val`

```scala
case class Message(sender: String, recipient: String, body: String)
val message1 = Message("guillaume@quebec.ca", "jorge@catalonia.es", "Ça va ?")

println(message1.sender)  // prints guillaume@quebec.ca
message1.sender = "travis@washington.us"  // this line does not compile
```

你不能给`message1.sender`重新赋值，因为它是一个`val`（不可变）。在案例类中使用`var`也是可以的，但并不推荐这样。

## 比较

案例类在比较的时候是按值比较而非按引用比较：

```scala
case class Message(sender: String, recipient: String, body: String)

val message2 = Message("jorge@catalonia.es", "guillaume@quebec.ca", "Com va?")
val message3 = Message("jorge@catalonia.es", "guillaume@quebec.ca", "Com va?")
val messagesAreTheSame = message2 == message3  // true
```

尽管`message2`和`message3`引用不同的对象，但是他们的值是相等的，所以`message2 == message3`为`true`。

## 拷贝

你可以通过`copy`方法创建一个案例类实例的浅拷贝，同时可以指定构造参数来做一些改变。

```scala
case class Message(sender: String, recipient: String, body: String)
val message4 = Message("julien@bretagne.fr", "travis@washington.us", "Me zo o komz gant ma amezeg")
val message5 = message4.copy(sender = message4.recipient, recipient = "claire@bourgogne.fr")
message5.sender  // travis@washington.us
message5.recipient // claire@bourgogne.fr
message5.body  // "Me zo o komz gant ma amezeg"
```

上述代码指定`message4`的`recipient`作为`message5`的`sender`，指定`message5`的`recipient`为”claire@bourgogne.fr”，而`message4`的`body`则是直接拷贝作为`message5`的`body`了。



# 模式匹配



模式匹配是检查某个值（value）是否匹配某一个模式的机制，一个成功的匹配同时会将匹配值解构为其组成部分。它是Java中的`switch`语句的升级版，同样可以用于替代一系列的 if/else 语句。

## 语法

一个模式匹配语句包括一个待匹配的值，`match`关键字，以及至少一个`case`语句。

```scala
import scala.util.Random

val x: Int = Random.nextInt(10)

x match {
  case 0 => "zero"
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}
```

上述代码中的`val x`是一个0到10之间的随机整数，将它放在`match`运算符的左侧对其进行模式匹配，`match`的右侧是包含4条`case`的表达式，其中最后一个`case _`表示匹配其余所有情况，在这里就是其他可能的整型值。

`match`表达式具有一个结果值

```scala
def matchTest(x: Int): String = x match {
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}
matchTest(3)  // other
matchTest(1)  // one
```

这个`match`表达式是String类型的，因为所有的情况（case）均返回String，所以`matchTest`函数的返回值是String类型。

## 案例类（case classes）的匹配

案例类非常适合用于模式匹配。

```scala
abstract class Notification

case class Email(sender: String, title: String, body: String) extends Notification

case class SMS(caller: String, message: String) extends Notification

case class VoiceRecording(contactName: String, link: String) extends Notification
```

`Notification` 是一个虚基类，它有三个具体的子类`Email`, `SMS`和`VoiceRecording`，我们可以在这些案例类(Case Class)上像这样使用模式匹配：

```scala
def showNotification(notification: Notification): String = {
  notification match {
    case Email(sender, title, _) =>
      s"You got an email from $sender with title: $title"
    case SMS(number, message) =>
      s"You got an SMS from $number! Message: $message"
    case VoiceRecording(name, link) =>
      s"you received a Voice Recording from $name! Click the link to hear it: $link"
  }
}
val someSms = SMS("12345", "Are you there?")
val someVoiceRecording = VoiceRecording("Tom", "voicerecording.org/id/123")

println(showNotification(someSms))  // prints You got an SMS from 12345! Message: Are you there?

println(showNotification(someVoiceRecording))  // you received a Voice Recording from Tom! Click the link to hear it: voicerecording.org/id/123
```

`showNotification`函数接受一个抽象类`Notification`对象作为输入参数，然后匹配其具体类型。（也就是判断它是一个`Email`，`SMS`，还是`VoiceRecording`）。在`case Email(sender, title, _)`中，对象的`sender`和`title`属性在返回值中被使用，而`body`属性则被忽略，故使用`_`代替。

## 模式守卫（Pattern gaurds）

为了让匹配更加具体，可以使用模式守卫，也就是在模式后面加上`if <boolean expression>`。

```scala
def showImportantNotification(notification: Notification, importantPeopleInfo: Seq[String]): String = {
  notification match {
    case Email(sender, _, _) if importantPeopleInfo.contains(sender) =>
      "You got an email from special someone!"
    case SMS(number, _) if importantPeopleInfo.contains(number) =>
      "You got an SMS from special someone!"
    case other =>
      showNotification(other) // nothing special, delegate to our original showNotification function
  }
}

val importantPeopleInfo = Seq("867-5309", "jenny@gmail.com")

val someSms = SMS("867-5309", "Are you there?")
val someVoiceRecording = VoiceRecording("Tom", "voicerecording.org/id/123")
val importantEmail = Email("jenny@gmail.com", "Drinks tonight?", "I'm free after 5!")
val importantSms = SMS("867-5309", "I'm here! Where are you?")

println(showImportantNotification(someSms, importantPeopleInfo))
println(showImportantNotification(someVoiceRecording, importantPeopleInfo))
println(showImportantNotification(importantEmail, importantPeopleInfo))
println(showImportantNotification(importantSms, importantPeopleInfo))
```

在`case Email(sender, _, _) if importantPeopleInfo.contains(sender)`中，除了要求`notification`是`Email`类型外，还需要`sender`在重要人物列表`importantPeopleInfo`中，才会匹配到该模式。

## 仅匹配类型

也可以仅匹配类型，如下所示：

```scala
abstract class Device
case class Phone(model: String) extends Device {
  def screenOff = "Turning screen off"
}
case class Computer(model: String) extends Device {
  def screenSaverOn = "Turning screen saver on..."
}

def goIdle(device: Device) = device match {
  case p: Phone => p.screenOff
  case c: Computer => c.screenSaverOn
}
```

当不同类型对象需要调用不同方法时，仅匹配类型的模式非常有用，如上代码中`goIdle`函数对不同类型的`Device`有着不同的表现。一般使用类型的首字母作为`case`的标识符，例如上述代码中的`p`和`c`，这是一种惯例。

## 密封类

特质（trait）和类（class）可以用`sealed`标记为密封的，这意味着其所有子类都必须与之定义在相同文件中，从而保证所有子类型都是已知的。

```scala
sealed abstract class Furniture
case class Couch() extends Furniture
case class Chair() extends Furniture

def findPlaceToSit(piece: Furniture): String = piece match {
  case a: Couch => "Lie on the couch"
  case b: Chair => "Sit on the chair"
}
```

这对于模式匹配很有用，因为我们不再需要一个匹配其他任意情况的`case`。

## 备注

Scala的模式匹配语句对于使用[案例类（case classes）](https://docs.scala-lang.org/zh-cn/tour/case-classes.html)表示的类型非常有用，同时也可以利用[提取器对象（extractor objects）](https://docs.scala-lang.org/zh-cn/tour/extractor-objects.html)中的`unapply`方法来定义非案例类对象的匹配。



# 单例对象

单例对象是一种特殊的类，有且只有一个实例。和惰性变量一样，单例对象是延迟创建的，当它第一次被使用时创建。

当对象定义于顶层时(即没有包含在其他类中)，单例对象只有一个实例。

当对象定义在一个类或方法中时，单例对象表现得和惰性变量一样。

## 定义一个单例对象

一个单例对象是就是一个值。单例对象的定义方式很像类，但是使用关键字 `object`：

```
object Box
```

下面例子中的单例对象包含一个方法：

```
package logging

object Logger {
  def info(message: String): Unit = println(s"INFO: $message")
}
```

方法 `info` 可以在程序中的任何地方被引用。像这样创建功能性方法是单例对象的一种常见用法。

下面让我们来看看如何在另外一个包中使用 `info` 方法：

```
import logging.Logger.info

class Project(name: String, daysToComplete: Int)

class Test {
  val project1 = new Project("TPS Reports", 1)
  val project2 = new Project("Website redesign", 5)
  info("Created projects")  // Prints "INFO: Created projects"
}
```

因为 import 语句 `import logging.Logger.info`，方法 `info` 在此处是可见的。

import语句要求被导入的标识具有一个“稳定路径”，一个单例对象由于全局唯一，所以具有稳定路径。

注意：如果一个 `object` 没定义在顶层而是定义在另一个类或者单例对象中，那么这个单例对象和其他类普通成员一样是“路径相关的”。这意味着有两种行为，`class Milk` 和 `class OrangeJuice`，一个类成员 `object NutritionInfo` “依赖”于包装它的实例，要么是牛奶要么是橙汁。 `milk.NutritionInfo` 则完全不同于`oj.NutritionInfo`。

## 伴生对象

当一个单例对象和某个类共享一个名称时，这个单例对象称为 *伴生对象*。 同理，这个类被称为是这个单例对象的伴生类。类和它的伴生对象可以互相访问其私有成员。使用伴生对象来定义那些在伴生类中不依赖于实例化对象而存在的成员变量或者方法。

```
import scala.math._

case class Circle(radius: Double) {
  import Circle._
  def area: Double = calculateArea(radius)
}

object Circle {
  private def calculateArea(radius: Double): Double = Pi * pow(radius, 2.0)
}

val circle1 = Circle(5.0)

circle1.area
```

这里的 `class Circle` 有一个成员 `area` 是和具体的实例化对象相关的，单例对象 `object Circle` 包含一个方法 `calculateArea` ，它在每一个实例化对象中都是可见的。

伴生对象也可以包含工厂方法：

```
class Email(val username: String, val domainName: String)

object Email {
  def fromString(emailString: String): Option[Email] = {
    emailString.split('@') match {
      case Array(a, b) => Some(new Email(a, b))
      case _ => None
    }
  }
}

val scalaCenterEmail = Email.fromString("scala.center@epfl.ch")
scalaCenterEmail match {
  case Some(email) => println(
    s"""Registered an email
       |Username: ${email.username}
       |Domain name: ${email.domainName}
     """)
  case None => println("Error: could not parse email")
}
```

伴生对象 `object Email` 包含有一个工厂方法 `fromString` 用来根据一个 String 创建 `Email` 实例。在这里我们返回的是 `Option[Email]` 以防有语法分析错误。

注意：类和它的伴生对象必须定义在同一个源文件里。如果需要在 REPL 里定义类和其伴生对象，需要将它们定义在同一行或者进入 `:paste` 模式。

## Java 程序员的注意事项

在 Java 中 `static` 成员对应于 Scala 中的伴生对象的普通成员。

在 Java 代码中调用伴生对象时，伴生对象的成员会被定义成伴生类中的 `static` 成员。这称为 *静态转发*。这种行为发生在当你自己没有定义一个伴生类时。
