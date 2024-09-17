# Linux 常用命令



## 杀死所有python 进程

```python
ps -ef | grep python| grep -v grep | awk '{print $2}' | xargs kill -9
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
