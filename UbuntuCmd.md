# Ubuntu常用命令

- [Ubuntu常用命令](#ubuntu常用命令)
  - [查询文件占用空间](#查询文件占用空间)
  - [进程管理](#进程管理)

## 查询文件占用空间

```bash
#查询文件夹大小, max-depth指定查询深度
du -h --max-depth=1

#查询文件大小
du -h --max-depth=0 *

#查询磁盘空间
df -hl
```

## 进程管理

```bash
#查询进程号
ps -ef | grep process_name

#杀死进程
kill -s 9 pid
```