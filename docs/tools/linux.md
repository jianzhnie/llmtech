# linux

Linux：是一种自由和开放源码的类 UNIX 操作系统。该操作系统的内核由林纳斯·托瓦兹在 1991 年 10 月 5 日首次发布，在加上用户空间的应用程序之后，成为 Linux 操作系统

## liunx 常用命令

```shell
# 查看用户列表
cat /etc/passwd 
# 强制杀死进程
kill -9 pid 
# 杀死 nginx 所有的进程
killall nginx 
# ssh 启动 
sudo service ssh start
# 修改私钥权限
Chmod 600 ubuntu02_key
# 查看 nginx 的进程
ps -ef | grep nginx
# 设置 root 用户密码
sudo passwd 
# 切换用户
su root 
# 环境变量配置文件
/etc/profile 
# 远程登录
ssh roottest@192.168.50.141
ssh root@192.168.50.14 -P 8090
# 复制文件：
scp width.html roottest@192.168.50.141:home/newtest
# 复制目录：
scp -r filename roottest@192.168.50.141:home/newtest
# 访问网络
curl 127.0.0.1
# 退出
Exit
# 更改主机用户名
hostname hp 
# 查找文件位置
where is nginx 
# (别名)查找文件位置
which nginx 
# 查看帮助
main ls 
# 查看历史记录
history 
# 产看系统信息
top
# 查看内存
free 
# ip 地址
ip a 
# 查看端口
telnet www.baidu.com 80
```

## 权限问题

```shell
ls -a # 查看文件

chmod -R u+x tiral # 给文件赋予所有者的执行权限
chmod -R u-x tiral # 给文件减去所有者的执行权限
chmod -R u=rwx tiral # 给文件赋予所有者的读写执行权限
chmod 755 trial

chown user1 trial # 给文件赋予所有者的 fuzhi
chgrp user1 trial
```

## 执行命令

```shell
ls : 罗列文件列表
    -a :显示所有文件，包括隐藏文件
    -l :显示详细信息
    -d :查看目录属性
    -h :人性化显示文件大小
    -i :显示incode
```

```shell
mkdir 创建文件
    -p: 递归创建文件
```

```shell
cd  切换目录
    cd ~  :进入当前用户的家目录
    cd .. :进入上一次目录
    cd -  :进入上次目录
```

```shell
cp  复制目录
     -r :复制目录
     -p :连带文件属性
     -d :如果源文件是连接文件 则复制链接属性
     -a :相当于 -pdr
```

## 文件和目录操作

### Folder Size 查看目录的大小

```shell
du -h --max-depth=1
```

### Count files 大量文件的个数统计（超过ls的限制时）

```shell
find -type f -name '*.mp4' | wc -l
```

### Split and merge files

```shell
split --bytes=1GB     /path/to/file.ext /path/to/file/prefix
cat prefix* >     file.ext
```

## 搜索

```shell
find /root -name install.log # 不区分大小写
find /root -name "*[cd]"
find /root -nouser # 没有所有者文件， sys proc 没有属于正常。外来文件
find /var/log  -mtime +10 # 10天前
find /var/log  -mtime -10 # 10天内
find . -size 25k # 查找文件大小是25K的文件
find . -size +25K # 查找文件大小是大于25K的文件
find /etc -size +20k -a size -50k # 查找etc/ 目录下 大于20k并且小于50k 的文件
find /etc -size +20k -a size -50k -exec ls lh {}\; # 查找etc/ 目录下 大于20k并且小于50k 的文件,并显示详细信息
```

## 压缩

```shell
# .zip 格式压缩
zip  压缩文件名 源文件
zip -r 压缩文件名 源目录
uzip  解压文件名
# .gz 格式压缩
gzip 源文件
gzip -c 源文件 压缩文件
gzip -r 目录
gzip -d 解压文件
gunzip 解压文件
# .bz2 格式压缩
bzip2 源文件 压缩.bz2格式 不保留源文件
bzip2 -k 源文件 压缩之后保留源文件
bzip2 -d 解压文件
bunzip2 解压文件
# .tar.bz2 格式
tar  -jcv -f filename.tar.bz2 被压缩的文件或目录
tar -jxv -f filename.tar.bz2 -C 欲解压到的目录
# .tar.gz
tar -zcvf 压缩名 源文件 压缩
tar -zxvf 压缩包名称    解压
# .tar 格式
tar -cvf filename.tar xxx 打包
tar -xvf filename.tar  解压x
# install p7zip
sudo apt install p7zip
# extract 7z file
p7zip -d something.7z
```

## 关机

```shell
shutdown -r now
shutdown -r 05:30 &
shutdown -c
init 0
poweroff
reboot
```

## 设置环境变量

```shell
export HD='hello Docker'
echo $PATH  查看系统环境变量
PATH = "$PATH":/root/sh 增加 PATH 变量的值
```
