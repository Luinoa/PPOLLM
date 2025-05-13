# PPOAPI README

# 简介

本项目是基于 PPO 的集推理、训练为一体的 API server 项目，该项目是为了解决在 Kea+ 项目中，原来的 LLM 引导策略过于简陋的问题而研发的 LLM 框架，我们先是在 Kea 中实现了初步的推理，之后，考虑到兼容性、泛用性和实际部署问题，我们将该 API 完全解耦开来，完善实现了整个推理训练框架；虽然该项目是为了解决 Kea 项目的 LLM 路径引导问题，但该项目本身完全没有指定使用场景，只需满足 API 的格式要求，该 API 可以处理任意强化学习场景。

我们的 Kea 仓库：[GitHub - Flora-Alex/Kea: Property-based Testing for Mobile GUI Apps](https://github.com/Flora-Alex/Kea)

---

# 部署流程

请确保您已经配置好 PyTorch 和 cuda 环境，并且根据当前机器配置，适当调整环境。

首先，请 git 我们的仓库，然后安装必要的库：

```bash
git clone https://github.com/Luinoa/PPOLLM.git
pip install -r requirements.txt
```

再根据 shell 文件夹下的脚本，写出时候你的环境和需求的脚本，在根目录下运行，即可开始使用，对于参数的具体意义，请根据 api_server.py 下的描述进行使用，这里不再赘述。

如：

```
#!/bin/bash
# This script is used to run the inference server with large model.

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4,5,6,7


python ./api_server.py \
-t \
-p 8001 \
--policy-minibatch-size 1 \
--model Qwen/Qwen3-8B \
--load-path weights/PPO
```

特别注意：

--policy-minibatch-size 内存极端敏感，谨慎调节。

--model 建议使用支持 Accelerate 框架多卡推理的 HuggingFace 模型，根据我们的实验，8B 模型需要约 50G 显存（至少 3 张 3090）进行微调。

部署过程中要是还缺什么，可以试试“缺啥补啥”。

---


# 原理

## 主要原理

请参考该文档，对于底层原理，我们并没有很大改动：

[路径引导策略设计说明](https://ncn4377u7k7w.feishu.cn/wiki/LiilwSSkpi8kytk9zOWceZPBn1e?from=from_copylink)

最重要的改动是原来的观察结果是要先在任务特化的框架里被处理成文字之后再进行推理，而我们为了保证泛用性，直接将 obs 改成了如下所示的字典：

```python
{
  "task_id": "string",
  "obs": {
    "prompt": "string",
    "question": "string",
    "action": ["option1", "option2", "option3"]
  }
}
```

## 对推理逻辑的改进

我们改进了 PPO 推理的原实现方法，原实现方法将每一个 prompt 和 action 进行拼接，如果记 M 为 prompt 数量，N 为 action 总数，其空间复杂度达到了 O(MN)，由于我们的可选动作较多，在推理、训练时总是出现 OOM 的现象。

为了解决这个问题，我们则是使用了 KV cache 先缓存 prompt，再处理 action，将我们的空间复杂度降到了 O(M+N)，大幅降低了内存使用，成功解决了频繁出现 OOM 的问题。

## RAG整合
在普通推理的基础上，我们设计了一种RAG推理方法，使用检索增强的观测（结合历史上下文）生成动作。我们在 PPO 的基础上，增加了一个 RAG 步进接口 `rag_step`，
该接口会在每次 step 之后，使用检索增强的观测（结合历史上下文）生成动作。

改方法还未充分测试。

---

另外，还有很多小的细节上的改进，这里不再赘述。

---

# API 接口文档

### 通用说明

- **任务流程**：`attach` → 多次 `step` / `rag_step` → 多次 `feedback` → `detach`
- **任务 ID**：所有交互接口需携带有效的 `task_id`

---

### `POST /attach`

**描述**：新建一个交互任务会话。

**返回值**：

```json
{
  "status": "attached",
  "task_id": "string"
}
```

---

### `POST /detach`

**描述**：关闭并销毁任务会话。

**请求参数**（Query 或 JSON）：

```json
{
  "task_id": "string"
}
```

**返回值**：

```json
{
  "status": "detached",
  "task_id": "string"
}
```

或错误：

```json
{
  "error": "Task ID not found"
}
```

---

### `POST /ask`

**描述**：请求代理是否准备好接收新的输入。

**请求参数**：

```json
{
  "task_id": "string"
}
```

**返回值**：
成功：

```json
{
  "status": "ok"
}
```

失败：

```json
{
  "error": "Invalid task ID"
}
```

---

### `POST /step`

**描述**：标准步进，使用原始观测值（obs）生成动作。

**请求体**（JSON）：

```json
{
  "task_id": "string",
  "obs": {
    "prompt": "string",
    "question": "string",
    "action": ["option1", "option2", "option3"]
  }
}
```

**返回值**：

```json
{
  "task_id": "string",
  "status": "ok",
  "action": 0
}
```

**注意：**

action 列表过大可能会导致推理时间延长、OOM，在请求时，请注意根据环境的具体情况控制 action 列表的长度。

---

### `POST /rag_step`

**描述**：RAG 步进，使用检索增强的观测（结合历史上下文）生成动作。

**请求体**（JSON）：

```json
{
  "task_id": "string",
  "obs": {
    "prompt": "string",
    "question": "string",
    "action": ["option1", "option2", "option3"]
  }
}
```

**返回值**：

```json
{
  "task_id": "string",
  "status": "ok",
  "action": 0
}
```

**注意：**

action 列表过大可能会导致推理时间延长、OOM，在请求时，请注意根据环境的具体情况控制 action 列表的长度。

---

### `POST /feedback`

**描述**：向 PPO agent 提交该步的反馈（reward 和是否结束）。

**请求体**（JSON）：

```json
{
  "task_id": "string",
  "reward": 1.0,
  "done": false
}
```

**返回值**：

```json
{
  "status": "ok"
}
```