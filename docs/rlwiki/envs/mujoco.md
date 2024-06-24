# Mujoco

## 仿真器

机器人仿真是机器人学科的一大核心内容。通过仿真可以快速、低成本地验证结构、控制等方面，是机械结构设计优化、研究控制算法的有效手段。机器人仿真器主要由两个引擎组成:物理引擎和渲染引擎。其中物理引擎负责对物理系统、物理规律的模拟，如物体受力加减速、碰撞接触摩擦、柔性物体变形等物理现象的建模；渲染引擎则将物理引擎模拟的结果在2D屏幕(或者3D的VR设备)尽可能逼真地展示出来。

事实上，机器人仿真器跟游戏引擎有很密切的关系。许多最初应用于游戏的物理引擎/渲染引擎后来也被广泛应用于机器人仿真领域，如 [Bullet](https://github.com/bulletphysics/bullet3.git) 物理引擎，基于虚幻游戏引擎的无人驾驶仿真器 [AirSim](https://github.com/microsoft/AirSim) 等。不过游戏中的物理实现逻辑跟机器人领域还是有本质区别。游戏中更多追求视觉上的真实，前沿成果主要集中在柔性材料视觉效果、[海量粒子的运动模拟](http://github.com/taichi-dev/taichi)上；机器人仿真目前大部分还处于刚体系统模拟，除了视觉效果还追求其他物理效果的精确仿真：

1. **传感器**: 可以认为现有传感器的精确仿真都是机器人物理引擎的目标，如多维力/力矩传感器、IMU、拉力、触觉传感器、激光、立体相机。
2. **驱动器**: 机器人的动力单元如电机、气缸、液压元件、人工肌肉。
3. **非视觉的物理量**: 如摩擦力(静摩擦、滑动摩擦、滚动摩擦、扭转摩擦)、局部接触变形、弹簧/阻尼等。

<video class="ztext-gif GifPlayer-gif2mp4 css-1xeqk96" src="https://vdn6.vzuu.com/SD/be90ac5a-c8e9-11eb-966b-da79cbcd4089.mp4?pkey=AAX8uS9dcm8OUMQFx7fetj3kFv27kd1dv6wBR4MUFTrwfsuJEVxSbzvEF_IAc31no5cblF6o5EHlQ9d4CEiXBTZ9&amp;c=avc.0.0&amp;f=mp4&amp;pu=078babd7&amp;bu=078babd7&amp;expiration=1719211653&amp;v=ks6" data-thumbnail="https://pic4.zhimg.com/v2-f865580bd1c10065fab0e7a1ed2869ff_b.jpg" poster="https://pic4.zhimg.com/v2-f865580bd1c10065fab0e7a1ed2869ff_b.jpg" data-size="normal" preload="metadata" loop="" playsinline=""></video>

> 基于虚幻引擎的AirSim仿真器

在保证仿真精度的同时，机器人仿真还追求尽可能高的模拟速度(游戏一般保证60hz以下的实时速度即可；而机器人仿真一般要求几百上千的超实时速度，特别是强化学习等对数据采样数量大的场合，瓶颈在于仿真效率)。

## 物理引擎

机器人物理引擎狭义上仅对刚体前向动力学的模拟器,如[ODE](http://www.ode.org)。这种物理引擎输入用户端控制量(如关节电机力矩、空间作用力)，对刚体系统碰撞、接触和力学进行解算，输出整个系统的加速度，再通过积分器得到下一个系统状态，相关算法可参考前向动力学。如[Gazebo](https://github.com/gazebosim/gz-sim.git)、[V-rep](https://github.com/robocomp)这些仿真器内置了多种常见物理引擎，可以让用户自行选择。

广义上的物理引擎还需要上述传感器、驱动器、视觉渲染等模块，此时物理引擎更倾向于一种轻量的仿真器。今天介绍的 [MuJoCo](https://github.com/google-deepmind/mujoco.git) 即属于这种广义物理引擎，可以将引擎模块用于其他仿真器中(如集成在[Unity](http://www.mujoco.org/book/unity.html))，也可以直接用物理引擎来做仿真，如[pybullet](http://pybullet.org/wordpress/)。

游戏市场对机器人仿真领域冲击很大。积极上说，游戏产业的发展加速了对物理现象仿真的研究，并且随着游戏物理引擎逼真度的提升，越来越多的功能被应用在机器人仿真领域。不过机器人仿真跟游戏还是有上述本质区别，最基本的从底层力学工具就已经完全区分开:正经的游戏物理引擎一般不过多考虑刚体系统的性能优化，采用简单的牛顿力学工具，有助于游戏开发者理解原理；机器人物理引擎则倾向于采用Featherstone的空间向量方法，保证尽可能高效的刚体动力学算法实现(如MuJoCo、[RaiSim](https://github.com/raisimTech/raisimLib.git))。

下图中展示了常见的物理引擎，关于其性能分析可见文献。其中ODE、Bullet、[DART](https://github.com/dartsim/dart.git) 消费级机器人仿真中应用广泛，[OpenSim](https://github.com/opensim-org/opensim-core.git) 用于人体肌肉组织仿真，MuJoCo主要应用于强化学习和最优化控制领域，RaiSim是ETH Robotic Systems Lab 开发的目前最新的物理引擎，应用于他们实验室开发的四足机器人ANYmal。Bullet、[PhysX](https://github.com/NVIDIA-Omniverse/PhysX.git)、Havok应用于游戏，Bullet 由于版权优势在一些机器人领域有替代 MuJoCo 趋势。[Flex](https://github.com/NVIDIAGameWorks/FleX.git)是NVIDIA开发的引擎,在流体模拟、NVIDIA自家的GPU加速方面有优势，NVIDIA内部研究机构也有尝试应用于机器人[2]。

<img src="https://picx.zhimg.com/80/v2-8e84d7e9d97b291681f5de4775f630c2_1440w.webp?source=d16d100b" alt="img" style="zoom: 33%;" />

> 常见物理引擎

最后谈一谈机器人物理引擎之后可能发展的方向。目前在机器人初现的是将物理引擎跟控制结合起来。如强化学习中默认环境转移是黑箱，即使对环境建模也只是通过采样方法得到一个环境近似模型(即基于模型强化学习)，并没有利用到物理建模的先验知识。最近的研究如将物理引擎变成可微分形式,将控制看成一个结合环境模型的优化问题，可以将控制和物理仿真结合在一起。另外就是减小 sim-to-real gap 。因为物理引擎内部已经建立了环境的系统模型，一些机器人标定、辨识的[底层工具箱](http://roboti.us/optico.html)完全可以整合在引擎中。

## MuJoCo  简介

MuJoCo 全称为Multi-Joint dynamics with Contact (代表接触式多关节动力学)，它是一个通用的物理引擎,  旨在促进机器人、生物力学、图形和动画、机器学习和其他需要快速准确地模拟与其环境相互作用的铰接结构(如多指灵巧手操作)的领域研究和开发。 它最初主要由华盛顿大学的Emo Todorov 教授开发，后成立商业公司 Roboti LLC 来进行开发维护，于 2021 年 2022 月被 DeepMind 收购并免费提供，并于2022 年 5 月开源。MuJoCo 代码库可在GitHub上的deepmind/mujoco存储库中找到。

MuJoCo是一个带有C API的C / C++库，面向研究人员和开发人员。运行时模拟模块被调整为以最大限度地提高性能，并对由内置 XML 解析器和编译器预先分配的低级数据结构进行操作。不同于其他引擎采用 urdf 或者 sdf 等机器人模型， MuJoCo 引擎团队自己开发了一种机器人的建模格式MJCF （一种方便人们读写的XML 文件格式的语言），来支持更多的环境参数配置。该库包括用 OpenGL 呈现的带有本地 GUI 的交互式可视化界面。同时MuJoCo 进一步公开了大量用于计算物理相关量的高效函数。

MujoCo 可用于实现基于模型的计算，例如控制合成、状态估计、系统辨识、机制设计、通过逆动力学进行数据分析，以及机器学习应用的并行采样。它还可以用作更传统的模拟器，包括用于游戏和交互式虚拟环境。

## MuJoCo特征

Mujoco具有很多功能和特色，这里对一些出色的方面进行阐述：

### 1.广义坐标与现代接触动力学相结合

物理引擎传统上分为两类。机器人学和生物力学引擎，都在广义或关节坐标上使用高效精确的递归算法。然而，他们要么忽略了接触动力学，要么依赖于早期的需要很小时间采样的弹簧阻尼器方法。游戏引擎则使用了一种更现代的方法，通过解决一个最优化问题来发现接触力。然而，他们往往诉诸于过度规定的笛卡尔表示，其中关节约束是数值强加的，这使得涉及复杂的运动学结构时，造成不准确和不稳定的结果。MuJoCo 是第一个将两者结合起来的通用引擎: 广义坐标模拟和基于优化的接触动力学。其他模拟器最近也被改造成使用 MuJoCo 的方法，但这通常不能兼容它们的所有功能，因为它们从一开始就不是为此而设计的。习惯于游戏引擎的用户一开始可能会发现这个广义坐标有违直觉，后续将详细阐述。

- 游戏引擎(如ODE、Bullet、Physx)一般通过数值优化的方法来处理关节约束，会造成多刚体系统的不稳定和不精确。
- MuJoCo则采用广义坐标系和基于优化方法的接触力学方法。

### 2.软、凸和解析可逆接触动力学

在现代接触动力学方法中，摩擦接触引起的力或冲量通常被定义为线性或非线性互补问题(LCP 或 NCP)的解，这两个问题都是 NP 难的。MuJoCo 是基于接触物理学的一个不同的公式，这个公式可以归结为一个凸最优化问题。我们的模型允许软接触和其他约束，并具有唯一定义的逆向数据分析和控制应用程序。提供优化算法的选择，包括推广到Gauss-Seidel方法，可以处理椭圆形摩擦锥。该求解器提供了摩擦接触的统一处理，包括扭转和滚动摩擦，无摩擦接触，关节和筋极限，干摩擦关节和筋，以及各种等式约束。

- 现代物理引擎大多数求解线性互补问题来处理约束，MuJoCo允许软体接触和其他约束，并包含一个独有的逆动力学模型来做数据分析。提出了新的摩擦力模型，支持滚动摩擦、扭转摩擦等多种摩擦力仿真。

### 3. 肌腱几何学

MuJoCo 可以建立肌腱的三维几何模型，它是服从包裹和通过点约束的最小路径长度的字符串。该机制类似于 OpenSim 中的机制，但是实现了一组更受限制的封闭形式的包装选项，以加快计算速度。它还提供机器人特有的结构，如滑轮和耦合自由度。肌腱可以用来驱动，也可以用来对肌腱长度施加不平等或相等的约束。

### 4. 通用驱动模型

在使用与模型无关的 API 时，设计一个足够丰富的驱动模型是具有挑战性的。MuJoCo 通过采用一个抽象的驱动模型来实现这个目标，该模型可以具有不同类型的传输、力的产生和内部动力学(即，使整体动力学成为三阶的状态变量)。这些组件可以实例化，以便以一种统一的方式建模马达、气动和液压缸、 PD 控制器、生物肌肉和许多其他执行器。

### 5. 可重构计算流程

MuJoCo 有一个顶级步进函数 mj _ step，它运行整个向前动态并提升模拟的状态。然而，在许多仿真以外的应用中，能够运行计算流程的选定部分是有益的。为此，MuJoCo 提供了大量可以任意组合设置的标志，允许用户根据需要重新配置流程，而不仅仅是通过选项选择算法和算法参数。此外，还可以直接调用许多低级函数。用户定义的回调可以实现自定义力场、执行器、碰撞例程和反馈控制器。

### 6.模型编译

如上所述，用户以名为 MJCF 的 XML 文件格式定义 MuJoCo 模型，相比URDF模型具有易读性、灵活配置等优点。 然后，内置编译器将该模型编译为低级数据结构 mjModel，该结构为运行时计算进行了交叉索引和优化。编译后的模型也可以保存在二进制 MJB 文件中，提高加载速度。

### 7. 模型与数据的分离

MuJoCo 在运行时将模拟参数分成两个数据结构(C 结构) :

- MjModel 包含模型描述，并且预计将保持不变。还有其他结构嵌入其中，包含模拟和可视化选项，这些选项需要偶尔更改，但这是由用户完成的。
- MjData 包含所有动态变量和中间结果。它被用作一个记事本，其中所有函数读取它们的输入并写入它们的输出——然后这些输出成为模拟流程中后续阶段的输入。它还包含一个预分配的和内部管理的堆栈，因此运行时模块在初始化模型之后不需要调用内存分配函数。

MjModel 由编译器构造。给定 mjModel，mjData 是在运行时构造的。这种分离使得模拟多个模型以及每个模型的多个状态和控件变得非常容易，从而促进了用于采样和有限差异的多线程处理。顶级 API 函数反映了这种基本的分离，并具有以下格式:

```cpp
void mj_step(const mjModel* m, mjData* d);
```

### 8. 互动式模拟与可视化

本地3D 可视化工具提供了网格和几何图形的渲染、纹理、反射、阴影、雾、透明度、线框、天空盒、立体可视化(在支持四缓冲 OpenGL 的视频卡上)。该功能用于生成3D渲染，帮助用户洞察物理模拟，包括视觉辅助，如自动生成的模型骨架，等效惯性盒，接触位置和法线，接触力可以分为法线和切线组件，外部扰动力，局部框架，关节和执行器轴，以及文本标签。可视化工具需要一个具有 OpenGL 呈现上下文的通用窗口，从而允许用户采用自己选择的 GUI 库。通过 MuJoCo 发布的 simulate.cc 代码示例展示了如何使用 GLFW 库实现这一点。一个相关的可用性特征是能够“触及”模拟，推动对象周围，看看物理如何响应。用户选择要施加外力和扭矩的物体，并看到扰动及其动态后果的实时渲染。这可以用来可视化地调试模型，测试反馈控制器的响应，或者将模型配置为所需的姿态。

### 9. 强大而直观的建模语言

MuJoCo 有自己的建模语言叫做 MJCF 。MJCF 的目标是提供对 MuJoCo 所有计算能力的访问，同时使用户能够快速开发新模型并进行实验。这个目标的实现在很大程度上是由于一个广泛的默认设置机制，类似于内嵌在 HTML 中的层叠样式表(CSS)。虽然 MJCF 有许多元素和属性，但是在任何给定的模型中，用户需要设置的元素和属性少得惊人。这使得 MJCF 文件比许多其他格式更短、更易读。

### 10. 复合柔性对象的自动生成

MuJoCo 的软约束可以用来建模绳索、布料和可变形的3D 物体。这需要大量的规则的机体，关节，肌腱和约束一起工作。这个建模语言有高级的宏，模型编译器会自动将这些宏扩展为标准模型元素的必要集合。重要的是，这些产生的灵活对象能够与其余的模拟完全交互。

## MuJoCo 模型元素

本节提供可以包含在 MuJoCo 模型中的所有元素的简要描述。稍后我们将更详细地解释底层计算、在 MJCF 中指定元素的方式以及它们在 mjModel 中的表示。

### Options

每个模型都有下列三组Options。他们总是包括在内。如果未在 XML 文件中指定它们的值，则使用默认值。这些选项的设计使得用户可以在每个模拟时间步骤之前更改它们的值。但是，在一个时间步骤内，不应更改任何选项。

- mjOption

此结构定义了所有跟物理仿真相关的options，比如选择动力学算法、更改仿真流程(屏蔽一些不需要的步骤)、调节环境的system-level参数如中立加速度等。

- mjVisual

此结构包含所有可视化选项。还有其他 OpenGL 呈现选项，但是这些选项是依赖于会话的，并且不是模型的一部分。

- mjStatistic

该结构包含了编译器计算的模型的统计信息: 如机体重量、所占空间范围等。包含它是为了提供信息，也是因为可视化工具将其用于自动定标。

### Assets

Assets本身不是模型元素。Assets用于增加或者改变元素默认信息。每个assets可以被多个模型元素引用(通过名字来引用)。

- Mesh

MuJoCo 可以从 OBJ 文件和二进制 STL 加载三角面片网格。有缩放选项用于缩放网格大小。像 MeshLab 这样的软件可以用来从其他格式转换。注意MuJoC定义的STL文件不支持颜色，而是使用引用 geom 的材质属性对网格着色。相反，所有的空间属性都由网格数据决定。MuJoCo 支持法线和纹理坐标的 OBJ 和自定义二进制文件格式。Mesh也可以直接嵌入到 XML 中。

- Skin

Skinned meshes (skins)表面柔性可变性的网格。它们的顶点附着在刚体上(在这里称为bones) ，每个顶点可以属于多个bones，从而导致skin的平滑变形。skin是纯粹的可视化对象，并不影响物理仿真，但他们可以显着提高视觉现实性。皮肤可以从自定义二进制文件加载，或直接嵌入到 XML 中，类似于网格。当自动生成复合灵活对象时，模型编译器还为这些对象生成skin。

- Height field

高度场可以从PNG文件(内部会转化成灰度图)中读取。在碰撞检测和可视化中被自动三角化。因此在碰撞检测中极可能在单个geom中出现大量接触点，此时只取前16个接触点。高度场适合用于建模大型的(相对控制的机器人)地表面。

- Texture

表面纹理信息一般从PNG或者用户定义颜色中读取。可视化界面支持两种纹理映射:2D 和方块cube。2D映射用于平面和高度场，cube映射用于收缩包装(shrink-wrapping)的3D物体。cube的六面可以从单独的文件读取或者复合图像文件，并且只能通过material来引用。

- Material

材质用于控制几何形状geoms，sites和绳tendons的外观。材料外观的纹理映射跟OpenGL中的RGBA、高光、反射、散射等特性交互。

### 运动链

MuJoCo一般用于约束刚体的仿真。系统状态定义在关节广义坐标系中，刚体被组织成运动链。除了worldbody 外其他刚体都只有一个父节点，并且运动环结构的定义不被允许。如果想要定义闭环结构话，需要通过equalityequalityequality属性来定义环关节。

因此MuJoCo模型的骨架由一到多个运动链组成(自由浮动刚体也可以定义成一个运动链)，下面列出的元素依附于具体某个刚体(下一节中的元素则不依附任何刚体)。

- Body

刚体body有质量和惯量特性，但没有任何几何性质。每个刚体有两个坐标系:用于定义刚体和其他元素位姿关系的坐标系，和一个位于刚体质心并且对其主惯量轴的惯量坐标系(此坐标系中惯量矩阵是对角形式)。在每一步仿真中MuJoCo递推计算前向动力学，得到刚体在全局坐标系中的位姿。

- Joint

关节定义在刚体中，来创建刚体和其父节点刚体的自由度(当关节缺失时两个刚体固接)。有四种类型的关节:球关节ball, 滑动关节slide, 旋转关节hinge, 自由关节free。一个身体可以有多个关节。通过这种方式，可以自动创建复合关节，而无需定义虚拟主体。球和自由关节的方向分量表示为单位四元数，MuJoCo 中的所有计算都考虑了四元数的性质。

- DOF

自由度跟关节数相关，但不是一一对应，这是由于球关节和自由关节有多个自由度。自由度有速度相关的性质如摩擦损失、阻尼、电机惯量等。广义力在自由度空间内表达；关节具有位置相关的性质如范围限制、弹簧刚度等。自由度不直接由用户定义，而是通过编译器仿真指定关节时创建。

- Geom

Geom是刚性附着在身体上的3D 形状。一个刚体可以有多个geoms。geom可以用于碰撞检测、渲染、自动推断刚体质量惯量。MuJoCo支持多种基元几何形状: plane, sphere, capsule, ellipsoid, cylinder, box。Geom 也可以是Mesh或Height Field; 这是通过引用相应的Assets来实现的。geom有大量的材质相关特性用于仿真和可视化。

- Site

Site可以认为是零质量的geom。其定义了刚体坐标系中感兴趣的位置点，通常用于定义一些对象（如传感器、肌腱路径点、滑块终点）的位置。不参与碰撞检测。

- Camera

MuJoCo 默认存在一个相机，用户可以在可视化界面中自由移动鼠标来切换视角。另外用户也可以自己定义一些相机视角用于立体渲染等。

- Light

光源可以固接在世界坐标系下或者移动的刚体上。MuJoCo实现的可视化界面可以访问OpenGL中所有的光源模型如环境光、漫反射、镜面反射、衰减截止、方向光、雾、阴影等等。默认的主光源跟相机一起移动，其特性可以通过mjVisual选项来改变。

### 独立元素

在这里，我们描述的模型元素不属于任何刚体，可以在运动链外被定义。

- Reference pose

Reference pose是存储在 mjModel.qpos0中的关节位置的向量。当模型处于初始配置时，它对应于关节的数值。每当重置模拟时，节点配置 mjData.qpos 都被设置为 mjModel.qpos0。在运行时，节点位置矢量解释相对于参考姿态。特别是，连接应用的空间转换量是 mjData.qpos-mjModel.qpos0。这个转换是除了存储在 mjModel 的 body 元素中的父子转换和旋转偏移之外的。Ref 属性只适用于标量关节(slide和hinge)。对于球关节，保存在 mjModel.qpos0中的四元数总是(1,0,0,0) ，它对应于空旋转。对于自由关节，浮体的全局3D 位置和四元数保存在 mjModel.qpos0中。

- Spring reference pose

关节和线缆弹簧平衡状态下的位置。当远离平衡点时会产生正比位置变化量的弹簧力。弹簧的初始值存在mjModel.qpos_spring中。对于滑动关节和旋转关节，初始参考值通过属性springref指定。对于球关节和自由关节，参考值由初始模型配置确定。

- Tendon

Tendon是标量长度单元，可用于驱动、施加极限和等式约束，或创建弹簧阻尼器和摩擦损失。Tendon有两种类型: 固定型和空间型。固定Tendon是(标量)关节位置的线性组合。它们对于建立机械耦合模型很有用。空间Tendon被定义为通过一系列指定位置(或通过点)或包围指定geom的最短路径。仅支持球体和圆柱体作为包装geom，并且为了包装的目的，圆柱体被视为具有无限长度。为了避免Tendon从包装凸轮的一侧突然跳到另一侧，用户还可以指定首选侧。如果在肌腱路径中有多个包裹物，它们必须被位置分开，以避免迭代求解的需要。空间Tendon也可以用滑轮分成多个分支。

- Actuator

MuJoCo提供了灵活的驱动器模型，包含三个可以独立指定的组件:  传动机构(transmission)、驱动动力学(activation dynamics)、力生成机制(force generation)。传动系统指定了驱动器如何连接到(可通过关节、线缆、滑块)系统上。驱动动力学用于建模驱动器的内部激励状态转移(如非线性的气缸、液压缸、人工肌肉)，最高支持三阶系统。力生成机制定义了如何将输入的控制信号映射到标量的驱动力上，再转换到广义力(乘以传动机构的力臂)。

- Sensor

MuJoCo 可以生成保存在全局数组 mjData.sensordata 中的模拟传感器数据。结果不用于任何内部计算; 相反，提供结果是因为用户可能需要它进行自定义计算或数据分析。可用的传感器类型包括触摸传感器、惯性测量单元(IMU)、力矩传感器、关节和肌腱位置和速度传感器、执行器位置、速度和力传感器、运动捕获标记位置和四元数以及磁力计。其中一些需要额外的计算，而另一些则从 mjData 的相应字段复制。还有一个用户自定义传感器，允许用户代码在传感器数据阵列中插入任何其他感兴趣的数量。MuJoCo 还具有离屏渲染功能，可以直接模拟彩色和深度摄像头传感器。这不包含在标准传感器模型中，而是必须以编程方式完成，如代码示例 [simulate.cc](https://mujoco.readthedocs.io/en/stable/programming/samples.html%23sasimulate). 所示。

- Equality

等值约束可以在运动链中添加额外的约束，如创建闭环关节或者机械耦合。仿真时模型内部力将等值约束跟其他约束一起处理。实现的等值约束有:通过一个球关节点连接两个刚体、刚体固接、两平面滑动约束、固定线缆/关节的空间位置、通过三次多项式耦合两个关节/线缆的位置。

- Contact pair

合理的接触选择有助于提高仿真效率。MuJoCo提供了十分精细的接触生成过程。geom的接触检测过滤来自两个方面:两物体足够近时触发、dynamic过滤器；通过MJCF文件显式定义。接触涉及到两个geoms，MJCF显式定义方式有助于用户定义一些dynamic机制无法实现的功能，可用于微调接触模型。

- Contact exclude

接触筛除定义了哪些geoms接触需要被排除在外，从而避免一些不想要的接触。

- Custom numeric

在 MuJoCo 模拟中有三种输入自定义数字的方法。首先，可以在 XML 中定义全局数字字段。它们有一个名称和一个实值数组。其次，某些模型元素的定义可以通过特定于元素的自定义数组进行扩展。这是通过在 XML 元素大小中设置属性 nuser _ XXX 来完成的。第三个是数组 mjData.userdata，任何 MuJoCo 计算都不使用它。用户可以在那里存储自定义计算的结果; 回想一下，随时间变化的所有东西都应该存储在 mjData 中，而不是 mjModel 中。

- Custom text

可以在模型中保存自定义文本字段。它们可以用于自定义计算——或者指定关键字命令，或者提供一些其他文本信息。但是不要将它们用于注释; 在编译后的模型中保存注释没有任何好处。XML 有自己的注释机制(MuJoCo 的解析器和编译器忽略了这一点) ，这种机制更为合适。

- Custom tuple

Custom tuple是MuJoCo元素的列表，用于指定特定元素的集合来进行自定义计算。比如用户可以使用包含特定刚体配对的tuple来进行自己实现的碰撞检测过程。

- Keyframe

Keyframe是对仿真中状态变量的快照，包含关节位置、速度、驱动器驱动量、仿真时间。一般用于重置仿真到某个状态。注意keyframe不用于记录轨迹，轨迹需要记录到用户实现额外的外部文件中(避免占用太多内存)。

### 关于Bodies、geoms、sites的关系

Bodies、geoms、sites是 MuJoCo 元素，它们大致相当于物质世界中的刚体。那为什么它们是分开的呢？这里解释了语义和计算方面的原因。

首先是相似点。Bodies、geoms、sites都有附着在它们上面的空间框架(尽管物体也有第二个框架，这个框架以物体的质心为中心，与惯性的主轴对齐)。这些框架的位置和方向计算在每个时间步从 mjData.qpos 通过正向运动学。正向运动学的结果可以在 mjData 中获得，如body的 xpos、 xquat 和 xmat，geom的Geom _ xpos 和 geom _ xmat，site的 site _ xpos 和 site _ xmat。

现在说说不同之处。body被用来构造运动学树，并且是其他元素的容器，包括geom和site。body有一个空间框架，惯性属性，但没有性质相关的外观或碰撞几何。这是因为这些属性不影响物理(当然除了接触，但这些是分开处理的)。如果你在机器人学教科书中看到过运动学树的图表，那么这些物体通常被画成无定形的形状——以表明它们的实际形状与物理学无关。

Geoms (几何基元的缩写)用于指定外观和碰撞几何。每个geom都属于一个body，并且与该物体紧密相连。多个geom可以连接到同一个body。MuJoCo 的碰撞检测器假设所有的几何图形都是凸的(如果网格不是凸的，它就用它们的凸壳在内部替换网格)。因此，如果你想建立一个非凸形状的模型，你必须把它分解成一个凸宝石的并，并把它们都附加到同一个body上。在 XML 模型中，Geoms 也可以有质量和惯性(或者更确切地说是用于计算质量和惯性的材料密度) ，但是这只用于计算模型编译器中的体质量和惯性。在实际的被模拟的 mjModel 中，geoms 没有惯性特性。

sites是light geom。它们具有相同的外观特征，但不能参与碰撞，不能用于推断物体质量。另一方面，site可以做一些 Geom 做不到的事情: 它们可以指定触摸传感器的体积、 IMU 传感器的附件、空间肌腱的路由、曲柄滑块驱动器的端点。这些都是空间量，但它们并不对应于应该有质量或与其他实体碰撞的实体——这就是为什么创建 site 元素。site还可以用来指定用户感兴趣的点(或帧)。

下面的示例说明了可以将多个sites和 geoms 连接到同一个body的一点: 在本例中，两个sites和 geoms 连接到一个body。

```xml
<mujoco>
  <worldbody>
    <body pos="0 0 0">
      <geom type="sphere" size=".1" rgba=".9 .9 .1 1"/>
      <geom type="capsule" pos="0 0 .1" size=".05 .1" rgba=".9 .9 .1 1"/>
      <site type="box" pos="0 -.1 .3" size=".02 .02 .02" rgba=".9 .1 .9 1"/>
      <site type="ellipsoid" pos="0 .1 .3" size=".02 .03 .04" rgba=".9 .1 .9 1"/>
    </body>
  </worldbody>
</mujoco>
```

## Mujoco 环境建模

### 模型实例

MuJoCo 中有几个称为“模型”的实体。用户在用 MJCF 或 URDF 编写的 XML 文件中定义模型。然后，软件可以在不同的媒体(文件或内存)和不同的描述级别(高或低)中创建同一模型的多个实例。如下表所示，所有组合都是可行的:

|        | High level           | Low level          |
| ------ | -------------------- | ------------------ |
| File   | MJCF/URDF (XML)      | MJB (binary)       |
| Memory | mjCModel (C++ class) | mjModel (C struct) |

所有运行时计算都是使用过于复杂而无法手动创建的方法执行的。

将模型分成high/low level的原因是:

1. 高层模型用于用户定义模型的便利,mjCModel数据结构基本上跟MJCF里面的定义一一对应，用户也可以直接编程来构建mjCModel。
2. mjCModel进一步编译成mjModel底层模型，底层模型直接用于计算和保存，不能被反编译恢复。

(内部) c++ 类大致与 MJCF 文件格式一对一关联。XML 解析器解释 MJCF 或 URDF 文件并创建相应的。原则上，用户可以通过编程方式创建，然后将其保存到 MJCF 或编译它。但是这个功能还没有公开，因为 C++ API 不能从独立于编译器的库中导出。我们计划围绕它开发一个 C 包装器，但目前解析器和编译器总是一起调用，而且模型只能在 XML.mjCModel中创建。

下图显示了获取mjModel的不同路径(第二个项目符号点尚不可用):

- (text editor) → MJCF/URDF file → (MuJoCo parser → mjCModel → MuJoCo compiler) → mjModel
- (user code) → mjCModel → (MuJoCo compiler) → mjModel
- MJB file → (MuJoCo loader) → mjModel

###  XML 文件

一个MuJoCo模拟器包含三部分：

- STL文件，即三维模型；
- XML 文件，用于定义运动学和动力学关系；
- 模拟器构建 Py 文件，使用mujoco-py将 XML model 创建成可交互的环境，供强化学习算法调用。

XML主要分为以下三个部分：

- `<asset>` ： 用`<mesh>` tag导入STL文件；
- `<worldbody>`：用`<body>` tag定义了所有的模拟器组件，包括灯光、地板以及你的机器人；
- `<acutator>`：定义可以执行运动的关节。定义的顺序需要按照运动学顺序来，比如多关节串联机器人以工具坐标附近的最后一个关节为joint0，依此类推。

下面是 MuJoCo 的 MJCF 格式的一个简单模型。它定义了一个固定在世界上的平面，一个可以更好地照亮物体和投射阴影的光线，以及一个具有6个自由度的浮动盒子(这就是“free”关节所做的)。

### 简单MJCF 模型定义实例

> 例1 下图是一个MJCF格式文件的简单实例。定义了一个固接在惯性坐标系中的平面、光照阴影和一个六自由度悬浮的方块。

```xml
<mujoco>
   <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
      <body pos="0 0 1">
         <joint type="free"/>
         <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
      </body>
   </worldbody>
</mujoco>
```

> 可视化的结果:
>
> <img src="https://pica.zhimg.com/80/v2-430579dc5859d6130ad7e2cd04c67f38_1440w.webp?source=d16d100b" alt="img" style="zoom:25%;" />

### 更复杂的MJCF格式模型定义

接下来，我们将提供一个更详细的示例，说明 MJCF 的几个特性。

```xml
<mujoco model="example">
    <compiler coordinate="global"/>
    <default>
        <geom rgba=".8 .6 .4 1"/>
    </default>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1"
                 width="256" height="256"/>
    </asset>
    <worldbody>
        <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
        <body>
            <geom type="capsule" fromto="0 0 1  0 0 0.6" size="0.06"/>
            <joint type="ball" pos="0 0 1"/>
            <body>
                <geom type="capsule" fromto="0 0 0.6  0.3 0 0.6" size="0.04"/>
                <joint type="hinge" pos="0 0 0.6" axis="0 1 0"/>
                <joint type="hinge" pos="0 0 0.6" axis="1 0 0"/>
                <body>
                    <geom type="ellipsoid" pos="0.4 0 0.6" size="0.1 0.08 0.02"/>
                    <site name="end1" pos="0.5 0 0.6" type="sphere" size="0.01"/>
                    <joint type="hinge" pos="0.3 0 0.6" axis="0 1 0"/>
                    <joint type="hinge" pos="0.3 0 0.6" axis="0 0 1"/>
                </body>
            </body>
        </body>
        <body>
            <geom type="cylinder" fromto="0.5 0 0.2  0.5 0 0" size="0.07"/>
            <site name="end2" pos="0.5 0 0.2" type="sphere" size="0.01"/>
            <joint type="free"/>
        </body>
    </worldbody>
    <tendon>
        <spatial limited="true" range="0 0.6" width="0.005">
            <site site="end1"/>
            <site site="end2"/>
        </spatial>asset
    </tendon>
</mujoco>
```

这个模型定义了一个七自由度的机械臂和长度受限的被动弹簧结构。字符串实现为具有长度限制的肌腱。肩部有球关节，肘部和腕部有成对的铰链关节。圆柱体内的盒子表示一个自由的“接头”。XML 中的最外层body 元素是必需的 worldbody。worldbody内部定义了机械臂末端通过一根柔性绳连接一个圆柱体。肩关节上是球关节，肘部和腕部各有两个正交的铰接关节。请注意，在两个主体之间使用多个关节并不需要创建虚拟主体。

用于动力学计算的刚体坐标系位姿、惯量从geoms几何形状中推算得到。site用于虚拟的定位点，这里被用来连接绳子的两端。default、asset分别定义了默认属性和背景属性等。可以看到MJCF格式相对URDF格式的建模文件可读性有了很大的提升。

MJCF 文件包含指定模型所需的最少信息。胶囊由空间中的线段定义——在这种情况下，只需要胶囊的半径。从所属的geoms中推断出机体的位置和方向。在均匀密度假设下，从geom形状推导出惯性性质。这两个站点被命名是因为肌腱定义需要引用它们，但是没有其他名称。关节轴仅定义为铰链关节，而不定义为球关节。冲突规则是自动定义的。摩擦特性、重力、模拟时间步长等设置为缺省值。顶部指定的默认 geom 颜色适用于所有 geom。

用于动力学计算的刚体坐标系位姿、惯量从geoms几何形状中推算得到。site用于虚拟的定位点，这里被用来连接绳子的两端。default、asset分别定义了默认属性和背景属性等。可以看到MJCF格式相对URDF格式的建模文件可读性有了很大的提升。

<img src="https://pic2.zhimg.com/v2-907bcebad7cc20cb1668b1f9e4a06251_b.jpg" alt="动图封面" style="zoom:50%;" />

## Mujoco Python 接口

### Dm-control

Dm-control是Deepmind开源的代码库，用于构建连续控制任务的强化学习场景，底层物理引擎使用的是MuJoCo，定义的场景模型也是MJCF格式。目前Dm-control代码库处于一个长期更新与维护的状态，最新版本已支持MuJoCo2.3.0。Dm-control的一大特点是设计任务与构建强化学习场景的标准化，开发人员也想将它作为强化学习算法解决连续控制问题的Benchmark，有许多常见的控制对象，例如倒立摆、仿生机器人以及机械臂等等。

Dm-control与Mujoco最常用的交互主要是封装在dm-control.mujoco.Physics这个类下，例如导入模型在dm-control里如下

```python
from dm_control import mujoco
physics = mujoco.Physics.from_xml_path('***.xml')
pixels = physics.render()
```

在Dm-control里，同样可以访问Mujoco的动静态仿真信息**mjModel**和**mjData**，直接通过Physics.model和Physics.data来访问，需要注意的一点是，由于储存数据的内存归Mujoco所有，所以直接尝试覆盖数组将会失败，利用python切片创建新对象的方式来赋值。

```python
import numpy as np
physics.data.qpos[:] = np.random.randn(physics.model.nq)
physics.data.ctrl[:] = np.random.randn(physics.model.nq)
```

同样的，Dm-control也支持状态重置，是以重置上下文的方式来保证各个信息的同步。

```python
with physics.reset_context():
    physics.data.qpos[:] = ... # Set position,
    physics.data.qvel[:] = ... # velocity
    physics.data.ctrl[:] = ... # and control.
```

###  Mujoco-py

OpenAI的 Mujoco-py 其实是openai的一个对接mujoco接口的项目, 目前最新版本支持 MuJoCo 2.1 的开源版本，代码库已经很久没有更新，而且mujoco_py 不支持 mujoco2.1.0 以上版本了。今后使用建议直接使用deepMind 官方给的python使用方案。

Mujoco-py的主要特点是支持实时GUI显示，GUI里有些现成的功能例如仿真加减速、接触力、逐帧播放和输出视频等功能，基本能涵盖仿真过程的基本要求。Mujoco-py早期还有并行运算API(MjSimPool)，后面项目人员觉得还是多进程做并行优势更大就给砍掉了。

### Mujoco Gym

Gym 提供了一套包含多种仿真环境的强化学习 API 标准接口，界面简单、Python 化，能够表示一般的 RL 问题，非常方便强化学习算法和仿真环境的对接。大家一定都非常了解下面这段强化学习入门代码：

```python
import gym
env = gym.make('Hopper-v4')
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminal, truncated, info = env.step()
    done =  terminal or truncated
    env.render()
    if done:
        obs, info = env.reset()
env.close()
```

但作为一个RL新手会非常好奇，这些状态、动作每个维度都是什么含义？reward、terminal 这些信息都是怎么定义的。想要深入了解如何在 Gym 中添加 Mujoco 模型，我们需要阅读 Gym 和Mujoco的说明文档以及 gym 以及 mujoco-py 的相关源码，下面介绍这背后的运行逻辑。

#### 状态

以Hopper为例，环境的状态组成，可以在 `Gymnasium/gymnasium/envs/mujoco/hopper_v4.py` 中的HopperEnv类找到_get_obs()方法。

```python
class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ...
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation
```

可以看到状态主要是两部分组成：`self.sim.data.qpos.flat `及`np.clip(self.sim.data.qvel.flat, -10, 10)`。可以从变量名字推断出，qpos指关节的位置（position），qvel指关节的速度信息（velosity）。

关节的位置、速度信息都是按照顺序给出的，关节的顺序可以通过`env.model.joint_names`或者`env.model._joint_id2name`获取。也可以简单粗暴地察看 `Gymnasium/gymnasium/envs/mujoco/assets/hopper.xml中的`joint`关键字。值得注意的是，一个关节的位置可能不止一个维度，如Ant-v2环境中名为`'root'`的位置信息便不止一维`env.data.get_joint_qpos('root')`。因此可以通过遍历关节的方式查看对应的索引。

```python
import gym
env = gym.make('Ant-v2)
for joint_name in env.model.joint_names:
    joint_addr = env.model.get_joint_qpos_addr(joint_name)
```

此外，也可以查看关节的range、vel、axis等信息，可以自行在MuJoCo中查找相关的接口，更简单的方法是看xml。

#### 动作

MuJoCo中的动作指关节的动作，同样可通过查看模型的xml文件中`<actuator>...</actuator>`中的内容获取，这是非常方便的办法。关节的顺序和前文提到的`env.model.joint_names`一致，需要注意的是并非所有的关节都会被执行动作，所以看xml是很方便的办法。

#### 奖励

以Hopper为例，可以直接看HopperEnv类中的step()方法如何计算奖励的。可以看出，奖励主要由三部分组成：前向速度、健康（存活）与否、动作消耗。需要注意的是，每个环境的奖励组成和比例都不同，Gym和rllab中同一个模型的奖励组成和比例也都可能不同。通常info信息也在此处给出。

```python
class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    ...
    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
```

#### Step

可以看到 HopperEnv 中的 Step 函数实际上是调用了`Mujoco` 中的 `mj_step ` 函数来完成的， 如下面的代码所示：

```python
class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)
```

#### MuJoCo模型

前面直接给出了查看状态、动作等信息的方法，但是为什么使用这些接口去查看？我们需要了解Gym是如何封装MuJoCo的，以及MuJoCo内部的信息是如何组成的。这里引用知乎一篇文章中的介绍：

> 一个MuJoCo模拟器是包含三部分的：
> STL文件，即三维模型；
> XML 文件，用于定义运动学和动力学关系；
> 模拟器构建py文件，使用mujoco-py将XML model创建成可交互的环境，供强化学习算 法调用。

其中，xml文件对应PyMjModel，模拟器对应 mujoco_py.  MjSim，模拟器数据对应PyMjData，此处的描述见[mujoco-py文档](https://openai.github.io/mujoco-py/build/html/reference.html%23)。在Gym中将这三者封装到`self.model`、以及`self.data`中。

```python
class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    self.model, self.data = self._initialize_simulation()

    def _initialize_simulation(
        self,
    ):
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data
```

`self.model`中包含模型相关的信息，`self.data`包含了模拟器运行过程中的相关数据，如状态等信息，可从[mujoco-py文档](https://openai.github.io/mujoco-py/build/html/reference.html%23)中查阅。

## MuJoCo 官方模型库

[mujoco_menagerie ](https://github.com/google-deepmind/mujoco_menagerie)是一个由 DeepMind MuJoCo 团队开发维护的存储库。该集合拥有一系列为 MuJoCo 物理引擎量身定制的高质量模型。
