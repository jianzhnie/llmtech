# Ubuntu下安装ZSH



## 安裝必要的套件

```shell
sudo apt install wget git curl vim -y
```

## 安裝 Zsh

shell 輸入

```shell
sudo apt install zsh -y
```

## 安裝 Oh My Zsh

輸入以下指令安裝 Oh My Zsh，安裝完畢後，按下 Enter 同意把預設 Shell 換成 Zsh。

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

国内的镜像源下载安装 oh-my-zsh

```shell
sh -c "$(curl -fsSL https://gitee.com/shmhlsy/oh-my-zsh-install.sh/raw/master/install.sh)" #国内镜像源
```

### 設定預設 Shell

若之前並沒有成功設定修改預設 Shell，請執行以下指令:

```shell
chsh -s $(which zsh)
```

執行 zsh 開始使用

```shell
zsh
```

## 安裝插件

安裝以下插件的時候，
請確定已安裝好 Oh My Zsh ，且目前正在使用的 Shell 是 Zsh。

### [主題 ](https://www.kwchang0831.dev/dev-env/ubuntu/oh-my-zsh#zhu3-ti2-ahrefhttpsgithubcomromkatvpowerlevel10k-relexternal-powerlevel10ka)[PowerLevel10k](https://github.com/romkatv/powerlevel10k)

```shell
git clone https://github.com/romkatv/powerlevel10k.git $ZSH_CUSTOM/themes/powerlevel10k
```

或者通过 git ssh

```shell
git clone git@github.com:romkatv/powerlevel10k.git $ZSH_CUSTOM/themes/powerlevel10k
```

### [插件 ](https://www.kwchang0831.dev/dev-env/ubuntu/oh-my-zsh#cha1-jian4-ahrefhttpsgithubcomzshuserszshautosuggestions-relexternal-zshautosuggestionsa)[zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)

```shell
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

或者通过 git ssh

```shell
git clone git@github.com:zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

### [插件 ](https://www.kwchang0831.dev/dev-env/ubuntu/oh-my-zsh#cha1-jian4-ahrefhttpsgithubcomzshuserszshsyntaxhighlighting-relexternal-zshsyntaxhighlightinga)[zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)

```shell
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

```shell
git clone git@github.com:zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

### [插件 ](https://www.kwchang0831.dev/dev-env/ubuntu/oh-my-zsh#cha1-jian4-ahrefhttpsgithubcomagkozakzshz-relexternal-zshza)[Zsh-z](https://github.com/agkozak/zsh-z)

類似於 [autojump](https://github.com/wting/autojump) 的插件，比 `cd` 更快速地直接跳到想去的資料夾，且效能更好沒有一堆依賴包。

```shell
git clone https://github.com/agkozak/zsh-z $ZSH_CUSTOM/plugins/zsh-z
```

或者通过 git ssh

```shell
git clone git@github.com:agkozak/zsh-z.git $ZSH_CUSTOM/plugins/zsh-z
```

