# neo4j操作命令

## Node语法

- 在cypher里面通过用一对小括号()表示一个节点，它在cypher里面查询形式如下：

~~~wiki
1，() 代表匹配任意一个节点
2, (node1) 代表匹配任意一个节点，并给它起了一个别名
3, (:Lable) 代表查询一个类型的数据
4, (person:Lable) 代表查询一个类型的数据，并给它起了一个别名
5, (person:Lable {name:"小王"}) 查询某个类型下，节点属性满足某个值的数据
6, (person:Lable {name:"小王",age:23})　节点的属性可以同时存在多个，是一个AND的关系
~~~

## 关系语法

- 关系用一对-组成，关系分有方向的进和出，如果是无方向就是进和出都查询

~~~shell
1,--> 指向一个节点
2,-[role]-> 给关系加个别名
3,-[:acted_in]-> 访问某一类关系
4,-[role:acted_in]-> 访问某一类关系，并加了别名
5,-[role:acted_in {roles:["neo","hadoop"]}]->
~~~

## 命令收藏

- 常用命令

~~~wiki
查询线路：MATCH (n:acline) RETURN n LIMIT 25
查询所有实体与关系：match (n) return n
清空实体关系：match (n) detach delete n
查询所有实体：match (n) return n
删除所有实体：match (n) delete n
查询所有关系：MATCH p=()-->() RETURN p
删除所有关系：MATCH p=()-->() delete p
~~~

- 创建

~~~shell
create (:Movie {title:"驴得水",released:2016})  return p;
~~~

- Merge a node and set properties if the node needs to be created.

~~~shell
MERGE (keanu:Person { name: 'Keanu Reeves' }) ON CREATE SET keanu.created = timestamp() RETURN keanu.name, keanu.created
~~~

- Merging nodes and setting properties on found nodes.

~~~shell
MERGE (person:Person) ON MATCH SET person.found = TRUE RETURN person.name, person.found
~~~

- 查最值

~~~shell
MATCH (p:acline)
RETURN max(p.attr_length)
~~~

- 筛选过滤

~~~shell
match (p1: Person) where p1.name="sun" return p1;
match (p1: Person {name:"sun"}) return p1; # 上一行与此等价
# 注意where条件里面支持 and ， or ，xor，not等boolean运算符，在json串里面都是and
# where里面查询还支持正则查询
match (p1: Person)-[r:friend]->(p2: Person) 
where p1.name=~"K.+" or p2.age=24 or "neo" in r.rels 
return p1,r,p2
~~~

- with语句给cypher提供了强大的pipeline能力，可以一个或者query的输出，或者下一个query的输入 和return语句非常类似，唯一不同的是，with的每一个结果，必须使用别名标识。

~~~shell
MATCH (person:Person)-[:ACTED_IN]->(m:Movie)
WITH person, count(*) AS appearances, collect(m.title) AS movies
WHERE appearances > 1
RETURN person.name, appearances, movies
~~~

- 结果集返回做去重

~~~shell
match (n) return distinct n.name;
~~~

- 排序和分页

~~~shell
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
RETURN a,count(*) AS appearances
ORDER BY appearances DESC SKIP 3 LIMIT 10;
~~~

## docker问题

- iptables问题：https://www.cnblogs.com/amoyzhu/p/9329368.html