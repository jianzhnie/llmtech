# Git

git 是一个分布式版本控制软件，最初由林纳斯·托瓦兹创作，于 2005 年以 GPL 发布。最初目的是为更好地管理 Linux 内核开发而设计。[Git](https://git-scm.com/) 是一个开源的分布式版本控制系统，可以有效、高速地处理从很小到非常大的项目版本管理。

## 优点

- 分布式开发，强调个体。
- 公共服务器压力和数据量都不会太大。
- 速度快、灵活。
- 任意两个开发者之间可以很容易的解决冲突。

## cheatsheet
![](tools/img/git-cheatsheet-cn.9c8eed56.jpeg)

<div align=center>
<img  src="tools/img/git-cheatsheet-cn.9c8eed56.jpeg"/>
</div>
<div align=center>图 1 git-cheatsheet</div>

## 常用命令

### 配置 config

- 查看信息

```shell
git config –list
```

- 配置邮箱和用户名

```shell
git config --global user.email xxx@xx.com
git config --global user.name "name"
```

- 查看用户名和邮箱

```shell
git config user.name  # 查看自己的用户名
git config user.email # 查看自己的邮箱地址
```

- Alias 简化命令

```shell
git config --global alias.a add // 添加别名 git add .
git config --global alias.co checkout
git config --global alias.ci commit
git config --global alias.br branch
git config --global alias.st status

# Then you can use "git co" instead of "git checkout", for example:
git co master
```

### 新建仓库

- 网页创建

```shell
git clone git@github.com:jianzhnie/deep_head_pose.git
cd deep_head_pose
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

- 本地创建仓库

```shell
##  create a new repository on the command line
echo "# deep_head_pose" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin git@github.com:jianzhnie/deep_head_pose.git
git push -u origin master
```

- 上传已经存在的仓库

```shell
## push an existing repository from the command line
git remote add origin git@github.com:jianzhnie/deep_head_pose.git
git push -u origin master
```

### 拉取代码

 - git  clone 操作

```shell
git clone https://github.com/user/test.git
```

- git push 操作

```shell
# 强制push 一般会丢失远程代码
git push -u https://gitee.com/user/test.git master -f
# 提交本地代码
git push origin master
# 删除远程仓库分支
git push origin --delete dev
```

- git remote 操作

```shell
# 查看关联的远程仓库的名称
git remote
# 查看关联的远程仓库的详细信息
git remote -v
# 删除远程仓库的关联
git remote remove <name>
# 修改远程仓库的关联地址
git remote set-url origin <newurl>
# 刷新获取远程分支
git remote update origin --prune
# 添加关联的远程仓库
git remote add go git@https://gitee.com/user/GO.git
```

- git pull 操作

```shell
git pull origin master 拉取本地代码
```

### 基本信息操作

```shell
# 将当前路径下修改文件添加至暂存区
git add .
# 查看状态
git status
git Untracked  未被追踪
# 修改未提交
git Modified
# 提交到代码区
git commit -m '修改代码'
# 撤销最近一次commit
git reset HEAD~
# 撤销版本
git reset --hard xxxx
# 比较当前文件和暂存区文件差异 git diff
git diff <file>
# 工作区和暂存区的比较
git diff HEAD -- <文件>
```

>  注意 --hard 参数会抛弃当前工作区的修改
>
> 使用 --soft 参数的话会回退到之前的版本，但是保留当前工作区的修改，可以重新提交

### 分支操作

```shell
# 查看本地分支
git branch
# 查看本地和远程分支 remotes开头的代表是远程分支
git branch -a
# 查看远程分支
git branch -r
# 创建dev分支
git branch dev
# 新建一个分支，并切换到该分支
git checkout -b [branch]
# 删除本地分支
git branch -d dev
# 新建一个分支，指向指定commit
git branch [branch] [commit]
# 分支重命名
git branch -m oldname newname
# 新建一个分支，与指定的远程分支建立追踪关系
git branch --track [branch] [remote-branch]
# 建立追踪关系，在现有分支与指定的远程分支之间
git branch --set-upstream [branch] [remote-branch]
# 删除远程分支
git push origin --delete dev
# 删除没有合并的分支
git branch -D test
# 修改分支名称
git branch –m dev fix
# 查看已经合并的分支
git branch --merged
# 查看已经合并的分支
git branch --no-merged
```

### 切换分支

```shell
# 恢复stage中的文件的工作区
git checkout .
# 取消本次修改，在工作区内
git checkout --
# 切换分支 dev
git checkout dev
# 创建并切换分支 dev
git checkout -b dev
# 恢复上次版本
git checkout a.tex.
# 拉取远程分支到本地 <本地分支名称> <远程分支名称>
git checkout -b D_1.3.0 origin/D_1.3.0
# 在本地创建和远程分支对应的分支
git checkout -b origin/
# (test 分支 向后移)
git rebase master
# 合并分支
git merge test
# 分支的某些 commit-hash
git cherry-pick dev-3.0
# merge 前的版本号
git reset --hard
# 撤销合并当前 merge
git revert -m merge
# 撤销指定的提交
git revert <commitd>
```

### git submodules

```shell
# Add a submodule to a repo
git submodule add <url> <name>
git add <name>
git commit -m "example comments"
git push

# Pull a repo with its submodules
git pull --recurse-submodules
```

### git ssh key

```shell
# 生成 ssh key
ssh-keygen -t rsa -C "your_email@example.com"
```

默认会在相应路径下（~/.ssh 文件夹）生成`id_rsa`和`id_rsa.pub`两个文件.

将ssh key添加到GitHub中, 文本编辑器打开`id_rsa.pub`文件，里面的信息即为SSH key，将这些信息复制到GitHub的`Add SSH key`页面即可.

添加后，在终端（Terminal）中输入, 验证是否添加成功。

```
ssh -T git@github.com
```

### 查看 log

```shell
# 查看log信息
git log –oneline
# 查看每次详细修改内容的diff
git log -p <file>
# 查看最近两次详细修改内容的diff
git log -p -2
# 查看log信息列表
git log --pretty=oneline
# 行内变化
git log -p --online
# 查看变化的文件
git log --name-only
# 查看文件变化
git log --name-status
# 显示每次提交的信息
git log --stat
# 显示某次提交的内容
git show <commitid>
# 查看文件的什么人修改的每行的变化信息
git blame style.less
# 显示所有提交记录，每条记录只显示一行
git log --pretty=oneline
# 显示某个文件的每个版本提交信息：提交日期，提交人员，版本号，提交备注（没有修改细节）
git whatchanged file
# 修改上次提交描述 本次提交并存到上次
git commit --amend
# 提交时显示所有的diff
git commit  -v
# //使用新的commit 提交替换上次commit
git commit --amend -m 'meggahe'
```

### 回退&撤销

```shell
# 撤销工作区操作
git checkout :
# 撤销工作区操作
git restore:
# 缓存区回到工作区
git reset  changefile
git restore --staged changefile
# 缓存区回到初始区
git checkout HEAD changefile
# 本地仓库回到缓存区
# 撤销commits
git reset --soft HEAD~1
# 本地仓库回到工作区
git reset HEAD~1
# 本地仓库回到初始化
git reset. --hard HEAD~1
# Discard all local uncommitted changes::warning:
git reset --hard
# Discard all local unpushed changes: :warning:
git reset --hard @{u}
```

### 标签 tag

```shell
# 创建一个标签，默认为HEAD当前分支添加标签
git tag v1.0
# 为版本号为e8b8ef6添加v2.0标签
git tag v2.0 e8b8ef6
# 6cb5a9e 为版本号， 为6cb5a9e添加带有说明的标签，-a指定标签名,-m指定说明文字
git tag -a v3.0 -m "version 0.2 released"
# 根据标签查看指定分支
git show v0.2
# 查看所有标签
git tag
# 删除v1.0标签
git tag -d v1.0
# 把v0.9标签推送到远程
git push origin v0.9
# 推送所有尚未推送到远程的本地标签
git push origin --tags
# 删除远程标签, 先删除本地标签，再删除远程标签
git tag -d v0.9
git push origin :refs/tags/v0.9
```

### 暂存区

```shell
# 放到暂存区
git stash
git stash list
# 恢复暂存
git stash apply
# 回复第一个
git stash apply stash{0}
# 恢复并且删除暂存区
git stash pop
# 删除暂存区
git stash drop stash{0}
```

### git cherry-pick

```shell
# 调减需合并的代码
git cherry-pick
```

## Git Sync

例如，我最近 fork 了 `mmdetection` 官方仓库到我的 github 地址， 修改了部分文件，并且 push 到我的 github 上。过了一段时间， `mmdetection` 官方仓库有了新的更新， 但是我 fork 的版本没有包含进来，因此我该如何保持我维护的 `mmdetection` 和官方版本保持同步？

> In your local clone of your forked repository, you can add the original GitHub repository as a "remote". ("Remotes" are like nicknames for the URLs of repositories - origin is one, for example.) Then you can fetch all the branches from that upstream repository, and rebase your work to continue working on the upstream version. In terms of commands that might look like:

### Syncing a fork

**Step 1**

```shell
git clone https://github.com/open-mmlab/mmdetection.git

## use git remote to see the origin url
git remote -v
origin	https://github.com/open-mmlab/mmdetection.git (fetch)
origin	https://github.com/open-mmlab/mmdetection.git (push)
```

**Step 2**

```shell
## second, change the origin url
git remote set-url origin https://github.com/apulis/ApulisVision.git

## make sure you have modified successful
git remote -v
origin	git@github.com:apulis/ApulisVision.git (fetch)
origin	git@github.com:apulis/ApulisVision.git (push)
```

**Step 3**
Before you can sync, you need to add a remote that points to the upstream repository. You may have done this when you originally forked.

```shell
# Add the remote, call it "upstream":

git remote add upstream https://github.com/open-mmlab/mmdetection.git

## make sure you have modified successful
git remote -v
origin	git@github.com:apulis/ApulisVision.git (fetch)
origin	git@github.com:apulis/ApulisVision.git (push)
upstream	https://github.com/open-mmlab/mmdetection.git (fetch)
upstream	https://github.com/open-mmlab/mmdetection.git (push)
```

###  Fetching

There are two steps required to sync your repository with the upstream: first you must fetch from the remote, then you must merge the desired branch into your local branch.

Fetching from the remote repository will bring in its branches and their respective commits. These are stored in your local repository under special branches.

```shell
git fetch upstream

# Grab the upstream remote's branches
remote: Counting objects: 75, done.
remote: Compressing objects: 100% (53/53), done.
remote: Total 62 (delta 27), reused 44 (delta 9)
Unpacking objects: 100% (62/62), done.
From https://github.com/open-mmlab/mmdetection
 * [new branch]      master     -> upstream/master
```

We now have the upstream's master branch stored in a local branch, upstream/master

```shell
git branch -va
* master                  22d2612 Merge remote-tracking branch 'upstream/master'
  remotes/origin/HEAD     -> origin/master
  remotes/origin/master   22d2612 Merge remote-tracking branch 'upstream/master'
  remotes/upstream/master 7f0c4d0 fix sampling result method typo (#3224)
```

###  Merging

Now that we have fetched the upstream repository, we want to merge its changes into our local branch. This will bring that branch into sync with the upstream, without losing our local changes.

```shell
$ git checkout master
# Check out our local master branch
Switched to branch 'master'

$ git merge upstream/master
# Merge upstream's master into our own
Updating a422352..5fdff0f
Fast-forward
 README                    |    9 -------
 README.md                 |    7 ++++++
 2 files changed, 7 insertions(+), 9 deletions(-)
 delete mode 100644 README
 create mode 100644 README.md
```

If your local branch didn't have any unique commits, git will instead perform a "fast-forward":

```shell
$ git merge upstream/master
Updating 34e91da..16c56ad
Fast-forward
 README.md                 |    5 +++--
 1 file changed, 3 insertions(+), 2 deletions(-)
```

## Git 钩子函数

和其它版本控制系统一样，Git 能在特定的重要动作发生时触发自定义脚本。 有两组这样的钩子：客户端的和服务器端的。 客户端钩子由诸如提交和合并这样的操作所调用，而服务器端钩子作用于诸如接收被推送的提交这样的联网操作。 你可以随心所欲地运用这些钩子。
也即绝大部分项目中的 .git/hooks，默认存在的都是示例，其名字都是以 .sample 结尾，如果你想启用它们，得先移除这个后缀。把一个正确命名且可执行的文件放入 Git 目录下的 hooks 子目录中，即可激活该钩子脚本

```shell
pre-commit # 钩子在键入提交信息前运行
prepare-commit-msg # 钩子在启动提交信息编辑器之前，默认信息被创建之后运行。
commit-msg # 钩子接收一个参数，此参数即上文提到的，存有当前提交信息的临时文件的路径
post-commit # 钩子在整个提交过程完成后运行。
post-applypatch # 运行于提交产生之后，是在 git am 运行期间最后被调用的钩子
pre-rebase # 钩子运行于变基之前，以非零值退出可以中止变基的过程。
post-rewrite # 钩子被那些会替换提交记录的命令调用，比如 git commit --amend 和 git rebase（不过不包括 git filter-branch）。
pre-push # 钩子会在 git push 运行期间， 更新了远程引用但尚未传送对象时被调用

# 服务器端钩子
update # 脚本和 pre-receive 脚本十分类似，不同之处在于它会为每一个准备更新的分支各运行一次
post-receive # 挂钩在整个过程完结以后运行，可以用来更新其他系统服务或者通知用户
pre-receive # 处理来自客户端的推送操作时，最先被调用的脚本是 pre-receive
```

## .gitignore

.gitignore 文件对其所在的目录及所在目录的全部子目录均有效。通过将.gitignore 文件添加到仓库，其他开发者更新该文件到本地仓库，以共享同一套忽略规则

```shell
# 以'#'开始的行，被视为注释.
# 忽略掉所有文件名是 index.txt的文件.
index.txt
# 忽略所有生成的 html文件,
*.html
# index.html是手工维护的，所以例外.
!index.html
# 忽略所有.o和 .a文件.
*.[oa]
```

### 配置语法：

- 以斜杠“/”开头表示目录；
- 以星号“*”通配多个字符；
- 以问号“?”通配单个字符
- 以方括号“[]”包含单个字符的匹配列表；
- 以叹号“!”表示不忽略(跟踪)匹配到的文件或目录；

## Git 插件

### pre-commit

pre-commit 能够防止不规范代码被 commit，没有 husky 这么全面，但是你可以接着安装 pre-push 等插件来防止对应的 git 操作

```bash
  npm install pre-commit --save-dev
```

```json
//package.json
 "scripts": {
    "test:jest": "jest ",
    "test:report": "jest  --coverage --coverageDirectory=testreport",
    "test-reportone": "jest --testResultsProcessor=jest-stare ",
    "test:docs": "node_modules/.bin/jsdoc -c jsdoc.json",
    "test": "jest  --coverage --coverageDirectory=testreport  --testResultsProcessor=jest-stare ",
    "precommit": "npm run jest"
  },
  "pre-commit": {
    "run": "test"
  },
```

## Git 私有库搭建

1.  [gogs](https://gogs.io/)
2.  [gitlab](https://about.gitlab.com/install/)
