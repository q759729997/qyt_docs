# RASA聊天机器人

- 参考资料：<http://rasachatbot.com/>
- unit订火车票：<https://ai.baidu.com/forum/topic/show/869808>

## 流程

- NLU（自然语言理解）：识别意图（intent）和槽位实体
- 对话管理(dialogue management)：action动作

~~~
template：回复模板
~~~

### 架构：

- 1. 收到消息并将其传递给解释器(Interpreter)，解释器将其转换为包含原始文本，意图和找到的任何实体的字典。这部分由NLU处理。 
- 2. 跟踪器(Tracker)是跟踪对话状态的对象。它接收新消息进入的信息。
 - 1. 策略(Policy)接收跟踪器的当前状态。 
 - 3. 该策略选择接下来采取的操作(action)。 
 - 4. 选择的操作由跟踪器记录。
 - 5. 响应被发送给用户。

### NLU

- 返回结果

~~~
{
  "intent": {
    "name": "mood_unhappy",
    "confidence": 0.999127209186554
  },
  "entities": [],
  "intent_ranking": [
    {
      "name": "mood_unhappy",
      "confidence": 0.999127209186554
    },
    {
      "name": "deny",
      "confidence": 0.000466885045170784
    }
  ],
  "text": "so bad"
}

~~~