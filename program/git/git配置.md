# git配置

## windows配置

- `C:\Users\Administrator\.gitconfig`

~~~wiki
[filter "lfs"]
	clean = git-lfs clean -- %f
	smudge = git-lfs smudge -- %f
	process = git-lfs filter-process
	required = true
[user]
	name = 759729997
	email = qiaoyongtian@qq.com
	mail = 759729997@qq.com
[gui]
	encoding = utf-8
[core]
	autocrlf = false
[credential]
	helper = store
~~~