# Ubuntu常用命令

- [Ubuntu常用命令](#ubuntu常用命令)
  - [查询文件占用空间](#查询文件占用空间)
  - [进程管理](#进程管理)
  - [Git](#git)
  - [关于`Windows`命令行](#关于windows命令行)

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

## Git

```bash
# init setting
git config --global user.name "username"
git config --global user.email email-address

# generate ssh key
ssh-keygen -t rsa -C email-address
ssh -T git@github.com

# init a repo
git init
git add *
git commit -m "add ..."
git remote add origin git@github.com:Komorebi660/[仓库名].git
git push -u origin master

# create new branch
git checkout -b xxx
git push origin xxx

# merge branch
git fetch origin main
git merge origin/main

# reduce .git size
git gc --prune=now
```

## 关于`Windows`命令行

`Windows`中有一些文件夹包含空格, 要想访问带有空格路径的可执行文件, 需要对带有空格的部分添加双引号:

```
PS: C:/Program Files/CMake/bin/cmake.exe --version
The term 'C:\Program' is not recognized as the name of a cmdlet, function, script file, or operable program.

PS: "C:/Program Files/CMake/bin/cmake.exe" --version
At line:1 char:42
+ "C:\Program Files\CMake\bin\cmake.exe" --version
+                                          ~~~~~~~
Unexpected token 'version' in expression or statement.

PS: C:/"Program Files"/CMake/bin/cmake.exe --version
cmake version 3.25.1

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```