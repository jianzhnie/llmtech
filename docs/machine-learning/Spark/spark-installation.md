# Spark Installation with Maven & Eclipse IDE



## 安装说明

目前存在多种安装Spark框架的方式。经过实验和比较，在Windows 10操作系统上通过Maven和Eclipse IDE来安装Spark框架较为方便。在Linux操作系统上，也推荐通过Maven来安装Spark框架。

### Maven & Eclipse IDE说明

Maven是一个主要为Java项目提供统一的编译系统，简化编译过程，提供项目依赖管理和项目版本管理的程序开发框架。

Eclipse IDE是一个可在Linux/macOS/Windows上运行的集成开发环境。它主要支持Java项目开发，同时也支持多种开发语言，以及Ant，Maven等项目编译框架。

#### 参考网站

- [Maven项目官方网站](https://maven.apache.org/index.html)
- [Eclipse项目官方网站](https://www.eclipse.org/)

## 安装过程

### JDK安装

JDK (Java Development Kit)是由Oracle维护的Java开发程序包。目前最新的版本是JDK 14。由于兼容性问题，目前依然有许多项目使用JDK 8。现在我们选择安装最新的稳定版即可。

### Eclipse IDE安装

Eclipse IDE安装要下载一个在线安装程序并运行。安装过程需要连接外网。如果需要在没有网络环境的计算机上安装Eclipse，可以在官网上选Download Package（在DOWNLOAD 64 BIT按钮下面）。

安装时选择第一项，Eclipse IDE for Java Developers

安装Eclipse IDE前最好先安装JDK，安装程序会自动搜索JDK的安装位置并进行设置，否则之后再来配置会比较麻烦。

### Maven安装

Eclipse IDE已经自动集成了Maven框架。

### Spark安装

我们使用Eclipse自带的Maven来安装Spark

#### 新建Maven项目

- 打开Eclipse以后，先新建一个Maven项目。选择菜单栏中的Flie -> New -> Project...，在弹出窗口中选择Maven -> Maven Project
- 在New Maven Project窗口中，勾选"Create a simple project (skip archetype selection)"（之后我们手动配置），点击Next按钮
- 配置这个新的Maven项目，可以选择方便自己记忆的名字。我的命名如下：
    - Group Id: me.spark.app
    - Artifact Id: mySparkApp
    - Version: 1.0
    - Packaging: jar
    - Name: playersStats
- 点击Finish按钮，至此完成新建Maven项目。你可以在左侧的Package Explorer里找到这个项目

#### 配置Maven依赖(安装Spark框架)

##### `pom.xml`文件说明
新建完Maven项目之后，可以在项目中找到一个名为`pom.xml`的文件。通过修改这个文件的内容，我们就可以利用强大的Maven框架解决许多依赖和编译问题。

以下是我的`pom.xml`文件：

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>me.spark.app</groupId>
  <artifactId>playersStats</artifactId>
  <version>1.0</version>
  <name>playersStats</name>
  <!-- FIXME change it to the project's website -->
  <url>http://www.example.com</url>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>14</maven.compiler.source>
    <maven.compiler.target>14</maven.compiler.target>
  </properties>

  <dependencies>
    <!-- https://mvnrepository.com/artifact/org.apache.maven.plugins/maven-assembly-plugin -->
	<dependency>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-assembly-plugin</artifactId>
      <version>3.3.0</version>
    </dependency>
  	<!-- https://mvnrepository.com/artifact/org.apache.spark/spark-core -->
  	<dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-core_2.12</artifactId>
      <version>3.0.0</version>
    </dependency>
    <!-- https://mvnrepository.com/artifact/org.apache.spark/spark-sql -->
	<dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-sql_2.12</artifactId>
      <version>3.0.0</version>
    </dependency>
  </dependencies>

  <build>
	<plugins>
  	  <plugin>
    	<artifactId>maven-assembly-plugin</artifactId>
    	<version>3.3.0</version>
    	<configuration>
          <descriptorRefs>
            <descriptorRef>jar-with-dependencies</descriptorRef>
          </descriptorRefs>
        <archive>
          <manifest>
            <mainClass>me.spark.app.playersStats.Main</mainClass>
          </manifest>
        </archive>
        </configuration>
        <executions>
      	  <execution>
        	<id>make-assembly</id> <!-- this is used for inheritance merges -->
        	<phase>package</phase> <!-- bind to the packaging phase -->
        	<goals>
          	  <goal>single</goal>
        	</goals>
      	  </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
</project>
```

几点说明：

- 一旦在Eclipse中保存`pom.xml`文件，Maven就会自动开始进行依赖包安装和配置。安装过程中会使Eclipse IDE有些许卡顿，此时尽量停止操作来避免程序崩溃

- 该项目所有的.java源文件的package我设置为me.spark.app.playersStats（其实可以自行选择，但最好和Maven项目属性保持一致）

- Properties为项目的基本配置，里面的`maven.compiler.source`和`maven.compiler.target`为JDK版本设置，应该与你安装的JDK版本保持一致（注意这里指的是正式版本号的前缀，JDK 14应该设置14，而JDK8应该设置1.8）
- Dependencies为项目的依赖包。`maven-assembly-plugin`用于编译，而`spark-core_2.12`和`spark-sql_2.12`则代表用Scala 2.12编译的Spark Core和Spark SQL框架。如果需要使用Spark框架的其他部分（比如MLlib）或者其他框架，则需要在这里添加相应的程序包，其相应的dependency配置可在[Maven Repository网站](https://mvnrepository.com/)上找到
- Build为项目的编译配置。这里使用`maven-assembly-plugin`在打包JAR过程中将依赖包也打进去，否则调用JAR包中依赖Spark框架的类会出现问题。

#### 配置Run指令

可以通过右键Package Explorer中项目名字 -> Run As -> Maven Build...(注意选后面有三个点的)来添加Run指令。

我设置的一些Run指令

- Name: playersStats-compile
    - Goals: clean compile assembly:single
- Name: playersStats-exec
    - Goals: exec:java -e
    - Parameters (通过Add...添加)
        - Parameter Name: exec.mainClass
        - Value: me.spark.app.playersStats.Main

#### 运行

在src/main/java中右键选择New -> Class并设置类名为Main，并在Main.java中的main函数中写好Spark测试程序，就可以通过先后通过compile和exec运行程序了。参考测试代码如下：

```java
// package...
// import org.apache.spark....
// import ...

public static void main(String[] args) throws Exception {
    SparkSession spark = SparkSession
        .builder()
	.appName("Java Spark SQL basic example")
	.config("spark.master", "local")
	.getOrCreate();

    Dataset<Row> df = spark.read()
        .option("header", "true")
	.option("inferSchema", "true")
	.csv("data/players_stats_by_season_full_details.csv");
    df.printSchema();
    df.select("Player").show();
    df.select(col("Player"), col("GP")).show();
    df.filter(col("GP").gt(75)).show();

    spark.stop();
}
```
