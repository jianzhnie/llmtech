# 使用 OpenAI Whisper 将 YouTube 转换为文本

[OpenAI 发布了一种新的语音识别模型，可在GitHub](https://github.com/openai/whisper)上免费开源获得。它的用途之一是将 YouTube 视频转换为文本。

在本文中，我们将讨论如何使用 OpenAI Whisper 使用 Python 和 FastAPI 将 YouTube 视频转录为文本。

## 安装包

首先，让我们安装我们需要的包。我们可以将我们需要的所有pip包存储在`requirements.txt`文件中。

```python
whisper
git+https://github.com/openai/whisper.git
pytube
fastapi
uvicorn
```

关于 pip 包要记住的一件事：已经有一个名为*whisper*的包，所以我们需要告诉 pip 我们想要来自 OpenAI 的*whisper*包，目前在 GitHub 上可用。

### 虚拟环境（可选）

您还可以使用`virtualenv`来为此创建独立的包，以便我们可以确保我们正在安装正确的包。您可以通过运行以下命令来执行此操作：

```bash
pip install virtualenv
```

安装完成后`virtualenv`，您可以使用`virtualenv`命令创建一个新的虚拟环境。

```bash
virutlenv youtext
```

这将创建一个名为当前工作目录的新目录`youtext`，其中包含 Python 可执行文件和 pip 包管理器的副本。

要开始使用虚拟环境，您需要激活它。在类 Unix 系统（例如 Linux 或 macOS）上，您可以通过运行以下命令来执行此操作：

```bash
source youtext/bin/activate
```

现在您可以使用此命令安装所有必需的软件包。

```bash
pip install -r requirements.txt
```

## 下载 YouTube 视频

我们需要做的第一件事是下载 YouTube 视频。我们已经有了执行此操作的软件包`pytube`，因此我们可以通过以下方式下载它。首先，创建一个`download.py`文件。

```py
import hashlib
from pytube import YouTube

def download_video(url):
    yt = YouTube(url)
    hash_file = hashlib.md5()
    hash_file.update(yt.title.encode())
    file_name = f'{hash_file.hexdigest()}.mp4'
    yt.streams.first().download("", file_name)

    return {
        "file_name": file_name,
        "title": yt.title
    }
```

在这里，我们在保存已下载的视频时使用 MD5 对文件名进行哈希处理。您可以使用视频标题作为文件名，但有时访问它有点棘手。因此，对于本文，我们将使用 MD5 文件名。

## 转录成视频

我们已经有了视频文件，现在我们可以使用 Whisper 包转录它。现在，创建一个名为的文件`transcribe.py`；您可以随意命名该文件，但现在，我们将使用这个名称。

要使用耳语将视频转录为文本，我们可以简单地加载模型并像这样转录它。

```py
model.transcribe(video["file_name"])
```

在这种情况下，我们使用`base.en`模型。有多种型号可供选择；选择最适合您需要的型号。[请参阅此处的](https://github.com/openai/whisper#available-models-and-languages)文档。

![耳语模型](https://ahmadrosid.com/images/whisper-models.png)

现在我们可以导入我们之前创建的下载模块并使用它来下载视频，然后用 whisper 包转录它。

```py
import whisper
import os
from download import download_video

model_name = "base.en"
model = whisper.load_model(model_name)

def transcribe(url):
    video = download_video(url)
    result = model.transcribe(video["file_name"])
    os.remove(video["file_name"])

    segments = []
    for item in result["segments"]:
        segments.append(format_item(item))

    return {
        "title": video["title"],
        "segments": segments
    }
```

使用 Whisper 转录视频有几个字段可用，但在这种情况下，我们只关心 和`start`字段`text`。我们将使用此数据来显示文本并稍后将其链接到 YouTube。所以当我们在 YouTube 上打开视频时，我们可以得到准确的词。

所以，这里是我们如何格式化耳语的结果。

```py
def format_item(item):
    return {
        "time": item["start"],
        "text": item["text"]
    }
```

## 服务于互联网

现在我们已经有了 YouTube 视频转录的核心功能，让我们制作一个 API，以便我们可以将其部署为 Web 应用程序，以便您可以将其货币化，并可能从中建立一个价值数十亿美元的初创公司。

为了实现这一点，我们将使用`FastAPI`HTML 和 JQuery 来构建 Web 服务器并创建客户端界面，以与我们将要创建的 API 进行交互。

```py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transcribe import transcribe

app = FastAPI()

@app.get("/")
def index():
    index = ""
    with open("static/index.html", "r") as f:
        index = f.read()
    return HTMLResponse(index)

@app.post("/api")
def api(url: str = Form()):
    data = transcribe(url)
    return {
        "url": url,
        "data": data
    }
```

我们在这里定义了两条路线：

- `/`：此路由是一个 GET 端点，它返回一个 HTML 响应，其中包含名为 static/index.html 的文件的内容。
- `/api`：此路由是一个 POST 端点，它从请求正文中获取一个 url 参数，并返回一个包含 url 和以 url 作为参数调用转录函数的结果的 JSON 对象。

和函数用 FastAPI 路由装饰器装饰，它指定端点的 HTTP 方法和路由`index`。`api`

`/`当访问索引路由时，我们还提供静态文件以提供用户界面。

```py
@app.get("/")
def index():
    index = ""
    with open("static/index.html", "r") as f:
        index = f.read()
    return HTMLResponse(index)
```

## HTML 用户界面

我们将使用 CDN TailwindCSS 来设计我们的 UI 和 JQuery 以与我们已经创建的 API 进行交互。

```html
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"
    integrity="sha512-STof4xm1wgkfm7heWqFJVn58Hm3EtS31XFaagaa8VMReCXAkQnJZ+jEy8PCC/iT18dFy95WcExNHFTqLyp72eQ=="
    crossorigin="anonymous" referrerpolicy="no-referrer"></script>
```

对于 UI，我们将仅提供标题表单输入和一个用于转录文本的显示容器，如下所示。

```html
<body>
    <div class="text-5xl font-extrabold max-w-3xl mx-auto p-12 text-center">
        <span class="bg-clip-text text-transparent bg-gradient-to-r from-pink-500 to-violet-500">
            YouText
        </span>
        <p class="text-2xl font-light text-gray-600">Convert YouTube video to Text.</p>
    </div>
    <form id="form" class="max-w-3xl mx-auto space-y-4 p-8">
        <input name="url" class="rounded-sm p-2 w-full border" placeholder="Type youtube url here..." />
        <button type="submit" class="text-white bg-violet-500 rounded-sm w-full py-2">Submit</button>
    </form>

    <div class="max-w-3xl mx-auto space-y-4 p-8 bg-gray-200 relative">
        <h2 id="title" class="font-semibold text-2xl"></h2>
        <div class="absolute top-2 right-2 hover:cursor-pointer" id="copy-text">
            <!-- SVG Icon, get it here: https://gist.github.com/ahmadrosid/73b006f9265a262ace151bbce3a2d7fb -->
        </div>
        <div id="result"></div>
    </div>
</body>
```

调用 API 并格式化我们页面的输出是使用 JQuery 完成的，就像这样。

```html
<script>
    $(document).ready(() => {
        $("#form").submit(function (e) {
            e.preventDefault();

            let formData = $(this).serialize()
            let req = $.post("/api", formData, (data) => {
                let strTemp = ""
                data.data.segments.forEach(item => {
                    strTemp += `<a class="hover:text-violet-600" href="${data.url}&t=${parseInt(item.time, 10).toFixed(0)}s">${item.text}</a>`
                    if (item.text.includes(".")) {
                        $("#result").append(`<p class="pb-2">${strTemp}</p>`);
                        strTemp = "";
                    }
                });

                if (strTemp !== "") {
                    $("#result").append(`<p>${strTemp}</p>`);
                }

                $("#title").append(data.data.title);

                $("#copy-text").on("click", function () {
                    let $input = $("<textarea>");
                    $("body").append($input);
                    let texts = data.data.segments.map(item => item.text).join("").trim();
                    $input.val(texts).select();
                    document.execCommand("copy");
                    $input.remove();
                });

            });
            req.fail((err) => {
                console.log(err);
            });
        })
    })
</script>
```

这是我们的用户界面的外观。

![你的文字](https://github.com/ahmadrosid/YouText/raw/main/YouText.png)

## 结论

OpenAI 发布了一种新的语音识别模型，可在 GitHub 上免费开源获得。它的用途之一是将 YouTube 视频转换为文本。

在本文中，我们讨论了如何使用 OpenAI Whisper 使用 Python 和 FastAPI 将 YouTube 视频转录为文本。我们安装了必要的包，下载了 YouTube 视频，使用 Whisper 将其转录为文本，并使用 FastAPI 将其提供给互联网。

我们还使用 TailwindCSS 和 JQuery 创建了一个 HTML 用户界面来与 API 交互。最后，我们测试了该应用程序，它按预期工作。[在此处](https://github.com/ahmadrosid/YouText)获取完整的源代码。

## Reference

- https://ahmadrosid.com/blog/youtube-transcriptioin-with-openai-whisper
