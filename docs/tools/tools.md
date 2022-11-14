# Tools 效率工具

## Screen

Create a virtual screen:

```shell
screen -S [name]
```

- Exit screen: `Ctrl+A, D`
- Scroll: `Ctrl+A, Escape`

## Search history

```shell
history | grep 'something'
```

## Find & kill a process 关闭残留进程

```shell
ps -elf | grep python
kill -9 pid
```

## Monitor GPU utilization 监督GPU使用率

```shell
watch -n 1 -d nvidia-smi
```

## Image & Video Operations 图片视频操作

### FFMPEG

Compress a video 视频压缩

```shell
ffmpeg -i [src] -r 25 -b 3.5M -ar 24000 -s 960x540 [dst]
```

### Image resolution 查看图片的分辨率信息

```shell
file [filename]
```

### Del small-size images 删除所有的小于10k的jpg图

```shell
find . -name "*.jpg" -type 'f' -size -10k -delete
```
