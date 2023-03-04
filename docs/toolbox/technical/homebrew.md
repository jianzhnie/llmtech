# homebrew

## 安装方式

```shell
/usr/bin/ruby -e "\$(curl -fsSL https# //raw.githubusercontent.com/Homebrew/install/master/install)" //安装
/usr/bin/ruby -e "\$(curl -fsSL https# //raw.githubusercontent.com/Homebrew/install/master/uninstall)" //卸载
```

## 添加第三方仓库

```shell
# 安装 git
brew install git
# 重新覆盖安装包
brew reinstall git
# 卸载 git
brew uninstall git
# 搜索 git
brew search git
# 列出已安装的软件
brew list
# 更新 brew
brew update
# 用浏览器打开 brew 的官方网站
brew home
# 显示软件信息
brew info git
# 显示包依赖
brew deps
# 更新所有
brew upgarde
# 更新指定包
brew upgarde git
# 清理不需要的版本极其安装包缓存
brew cleanup
# 清理指定包的旧版本
brew cleanup git
# 查看可清理的旧版本包，不执行实际操作
brew cleanup -n
# 检查有没有问题
brew doctor
```
