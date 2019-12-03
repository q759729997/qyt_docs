# neo4j使用Python进行操作

删除

~~~python
MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r  # 删除全部
~~~

连接操作


```python
neo4j_url = 'http://39.104.161.233:7474/'
neo4j_username = 'neo4j'
neo4j_password = 'kdneo4j'
```


```python
from py2neo import Graph
```


```python
graph = Graph(neo4j_url, auth=(neo4j_username, neo4j_password))
```

创建实体与关系节点


```python
from py2neo import Node
from py2neo import Relationship
```


```python
a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
ab = Relationship(a, 'KNOWS', b)
graph.create(ab)
```


```python
# 带有属性的节点，node默认会多次写入
e1 = Node('Person', name='诸葛亮')
e1['职业'] = '军师'
e1['出生'] = '东汉末年'
e1
```




    (:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})




```python
graph.create(e1)
```

批量创建


```python
from py2neo import Subgraph
```


```python
tx = graph.begin()
nodes=[]
line_no = 0
for line in ['诸葛亮'] * 10:
    oneNode = Node('Person', name=line)
    line_no += 1
    oneNode['序号'] = line_no
    nodes.append(oneNode)
nodes = Subgraph(nodes)
tx.create(nodes)
tx.commit()
```


```python
matcher = NodeMatcher(graph)
for node in matcher.match('Person', name='诸葛亮'):
    print(node)
```

    (_40:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})
    (_60:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})
    (_80:Person {name: '\u8bf8\u845b\u4eae', 序号: 3})
    (_81:Person {name: '\u8bf8\u845b\u4eae', 序号: 4})
    (_82:Person {name: '\u8bf8\u845b\u4eae', 序号: 7})
    (_83:Person {name: '\u8bf8\u845b\u4eae', 序号: 8})
    (_84:Person {name: '\u8bf8\u845b\u4eae', 序号: 5})
    (_85:Person {name: '\u8bf8\u845b\u4eae', 序号: 6})
    (_86:Person {name: '\u8bf8\u845b\u4eae', 序号: 2})
    (_87:Person {name: '\u8bf8\u845b\u4eae', 序号: 9})
    (_88:Person {name: '\u8bf8\u845b\u4eae', 序号: 10})
    (_89:Person {name: '\u8bf8\u845b\u4eae', 序号: 1})


关系插入


```python
# 带有属性的节点，node默认会多次写入
e1 = Node('Person', name='刘备')
e1['职业'] = '商贩'
e1['出生'] = '东汉末年'
e1
```




    (:Person {name: '\u5218\u5907', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u5546\u8d29'})




```python
graph.create(e1)
```


```python
matcher = NodeMatcher(graph)
e1 = matcher.match('Person', name='刘备').first()
print('e1:{}'.format(e1))
e2 = matcher.match('Person', name='诸葛亮').first()
print('e2:{}'.format(e2))
if e1 and e2:
    relationNode = Relationship(e1, '指挥', e2)
    print('relationNode:{}'.format(relationNode))
    graph.create(relationNode)
```

    e1:(_24:Person {name: '\u5218\u5907', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u5546\u8d29'})
    e2:(_40:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})
    relationNode:(刘备)-[:指挥 {}]->(诸葛亮)


查询


```python
from py2neo.matching import NodeMatcher
```


```python
# 存在数据查询
matcher = NodeMatcher(graph)
matcher.match('Person', name='Alice').first()
```




    (_22:Person {name: 'Alice'})




```python
# 不存在的数据查询
matcher = NodeMatcher(graph)
matcher.match('Person', name='Alicekhjgbkbk').first()
```


```python
# 带属性的实体查询
matcher = NodeMatcher(graph)
matcher.match('Person', name='诸葛亮').first()
```




    (_40:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})




```python
matcher = NodeMatcher(graph)
for node in matcher.match('Person', name='诸葛亮'):
    print(node)
```

    (_40:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})
    (_60:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})



```python
for node in graph.match(nodes=None, r_type=None, limit=None): # 找到所有的relationships
    print(node)
```

    (Alice)-[:KNOWS {}]->(Bob)
    (刘备)-[:指挥 {}]->(诸葛亮)


关系查询


```python
from py2neo import RelationshipMatcher
```


```python
matcher = RelationshipMatcher(graph)
matcher.match(r_type='指挥').first()
```




    (刘备)-[:指挥 {}]->(诸葛亮)




```python
find_rela = graph.run("""MATCH p=(node1: Person)-[r: `指挥`]->(node2: Person)
WHERE node1.name = '刘备' AND node2.name = '诸葛亮'
RETURN p""")
for i in find_rela:
    print(i)
```

    <Record p=(刘备)-[:指挥 {}]->(诸葛亮)>



```python
find_rela = graph.run("""MATCH p=(node1: Person)-[r: `指挥111`]->(node2: Person)
WHERE node1.name = '刘备' AND node2.name = '诸葛亮'
RETURN p""")
print(len(find_rela))
for i in find_rela:
    print(i)
```


```python
def check_relation_node(node1_name, node1_type, node2_name, node2_type, relation, graph: Graph):
    """检查关系节点是否存在.

    Args:
        node1_name: 节点1名称.
        node1_type: 节点1类型.
        node2_name: 节点2名称.
        node2_type: 节点2类型.
        relation: 关系类型.
        graph: neo4j实例化对象.

    Returns:
        布尔结果，存在返回true，不存在返回false.
    """
    find_relations = graph.run("""MATCH p=(node1: {})-[r: `{}`]->(node2: {})
    WHERE node1.name = '{}' AND node2.name = '{}'
    RETURN p""".format(node1_type, relation, node2_type, node1_name, node2_name))
    for i in find_relations:
        return True
    return False
```

Python对象操作关系查询


```python
node_matcher = NodeMatcher(graph)
relation_matcher = RelationshipMatcher(graph)
e1 = node_matcher.match('Person', name='刘备').first()
print('e1:{}'.format(e1))
e2 = node_matcher.match('Person', name='诸葛亮').first()
print('e2:{}'.format(e2))
if e1 and e2:
    print(relation_matcher.match(nodes=[e1, e2],r_type='指挥').first())
```

    e1:(_24:Person {name: '\u5218\u5907', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u5546\u8d29'})
    e2:(_40:Person {name: '\u8bf8\u845b\u4eae', 出生: '\u4e1c\u6c49\u672b\u5e74', 职业: '\u519b\u5e08'})
    (刘备)-[:指挥 {}]->(诸葛亮)



```python

```
