

#  常用的 Prompt

## 论文/博客 翻译

```python
任务描述：请将以下AI领域的英语论文或博客内容翻译成中文，确保翻译准确、专业，并符合中文技术文献的表达习惯。输出格式为Markdown。

翻译要求：

1. 专业性：
   - 保留原文的技术术语和概念，确保翻译后的内容在AI领域内准确无误。
   - 如果术语有通用的中文译法，请直接使用；如果没有，可以保留英文术语并附上简要解释。

2. 逻辑清晰：
   - 确保翻译后的内容逻辑清晰，语句通顺，符合中文技术文献的阅读习惯。

3. 数学公式：
   - 使用 `$$` 符号修饰数学公式：
     - 行内公式：`$公式$`
     - 独立公式：`$$公式$$`

4. 文化适应性：
   - 如果原文涉及文化特定的表达或背景知识，请适当调整以使其在中文语境中易于理解。

5. 格式一致性：
   - 保留原文的段落结构、标题、列表、公式等格式，确保翻译后的内容与原文结构一致。

6. 语气和风格：
   - 保持原文的学术性或技术性语气，如正式、严谨、客观等。

输出格式：
- 翻译后的内容应以Markdown格式输出，确保标题、段落、列表、公式等格式与原文一致。


请根据以上要求，将以下AI领域的英语论文或博客内容翻译成中文：

```

## 代码纠错

```markdown
请检查以下代码是否存在错误，并指出错误的具体位置、原因以及修复方法。代码是用[编程语言]编写的，涉及[相关技术或框架]。如果代码逻辑有问题或存在潜在的性能问题，也请一并指出并提供优化建议。

要求：

1. 逐行或逐段检查代码，指出语法错误、逻辑错误或运行时错误。
2. 解释每个错误的原因，并提供具体的修复方法。
3. 如果代码存在潜在的性能问题或可优化之处，请说明并提出改进建议。
4. 如果代码涉及特定技术或框架，请确保修复方法符合其最佳实践。

示例输出格式：

1. 错误1：
   - 位置：第X行。
   - 错误描述：具体错误现象（如语法错误、逻辑错误等）。
   - 原因分析：解释错误的原因。
   - 修复方法：提供具体的修复代码或建议。
2. 错误2：
   - 位置：第Y行。
   - 错误描述：具体错误现象。
   - 原因分析：解释错误的原因。
   - 修复方法：提供具体的修复代码或建议。
3. 优化建议：
   - 指出代码中可能存在的性能问题或可改进之处，并提供优化方案。
```



## 代码规范

````python
Please help me review, debug, and improve the following code. The code is written in [programming language] and involves [related technologies or frameworks, if any]. Your task includes:

1. Error Checking and Correction:
   - Identify and fix any syntax errors, logical errors, or runtime errors in the code.
   - Explain the cause of each error and provide the corrected version.
2. Code Refactoring:
   - Rewrite the code to improve its structure, readability, and maintainability.
   - Use meaningful variable and function names that reflect their purpose.
   - Group related functions or logic into classes, modules, or sections as appropriate.
   - Simplify complex logic or break it into smaller, reusable functions.
   - Follow best practices for [programming language] and ensure the code adheres to PEP 8 (if Python) or relevant style guidelines.
3. Documentation:
   - Add clear and concise documentation (e.g., docstrings) to describe the purpose of functions, classes, and modules.
   - Include details about parameters, return values, and any exceptions raised.
4. Comments:
   - Include inline comments to explain complex logic or important steps in the code.
   - Ensure comments are meaningful and not redundant.
5. Type Annotations:
   - Add type hints (e.g., using `Typing` in Python) to make the code more robust and easier to understand, enable static type checking.
   - Ensure type annotations are consistent and accurate.
6. Optimization Suggestions:
   - Identify any potential performance issues or inefficiencies and suggest improvements.
   - Recommend ways to make the code more Pythonic (if applicable) or idiomatic for the language.
7. Final Output:
   - Provide the corrected, refactored, and fully documented code with all the above enhancements.
   - Include a summary of changes made and explanations for key improvements.
````

## 代码提升

```python
Please help me analyze and improve the code, do following things:
1. Fix any errors
2. Add proper type hints
3. Add documentation and comments
4. Improve code structure and readability
5. Follow best practices and coding standards
```

## 代码解释：

```markdown
请详细解释以下代码的功能、逻辑和关键部分及潜在的优缺点。请逐步分析代码的执行流程，并说明每个重要函数、变量或模块的作用。如果有优化建议或潜在问题，也请一并指出。

要求：

1. 代码的整体逻辑和流程。
2. 说明代码中使用的关键算法或方法的实现原理。
3. 指出代码中可能存在的性能瓶颈、潜在错误或改进空间。
4. 如果代码涉及特定技术或框架，请简要说明其背景和作用。
5. 逐行或逐段解释代码的功能。
5. 提供代码的优化建议（如果有）。

示例输出格式：

1. 代码功能概述：简要描述代码的整体功能。
2. 关键算法/技术：解释代码中使用的核心算法或技术。
3. 特定技术或框架: 对于代码中涉及特定技术或框架，简要说明背景和作用。
4. 代码逐行/逐段分析：
   - 第X行：解释该行代码的作用。
   - 第Y行：说明该行代码的逻辑。
   - ...
5. 潜在问题与优化建议：指出代码中可能存在的问题，并提供改进建议。

```
