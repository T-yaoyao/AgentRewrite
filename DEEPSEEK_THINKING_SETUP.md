# DeepSeek 思考模式配置指南

## 🎯 概述

AgentRewrite 系统现已支持 DeepSeek V3.2 的思考模式，可以在 SQL 重写过程中显示详细的推理过程。

## ⚙️ 配置步骤

### 1. 环境变量配置

在 `config_file/.env` 文件中添加以下配置：

```bash
# DeepSeek V3.2 (阿里云百炼)
REWRITE_MODEL_API_KEY=sk-your-dashscope-api-key
REWRITE_MODEL=deepseek-v3.2
REWRITE_MODEL_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2. 获取 API Key

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 开通 DeepSeek 相关服务
3. 获取 API Key

### 3. 支持的模型

系统会自动为以下模型启用思考模式：
- `deepseek-v3.2`
- `deepseek-v3`
- 任何包含 `deepseek` 且包含 `v3` 或 `thinking` 的模型名

## 🔍 思考模式特性

### 输出格式

```
====================思考过程====================
[DeepSeek的推理过程，会显示详细的思考步骤]

====================完整回复====================
[最终的SQL重写结果]
```

### 在哪些地方使用

- **RewriteAgent**: SQL重写过程中的推理显示
- **DecisionAgent**: 优化决策时的推理过程
- **ReasoningAgent**: 规则选择时的推理过程

## 🔧 技术实现

### 自动检测

```python
# 系统会自动检测是否为DeepSeek思考模型
self.is_deepseek_thinking = "deepseek" in model.lower() and ("v3" in model or "thinking" in model)
```

### API调用

```python
# 自动添加思考模式参数
extra_body={"enable_thinking": True}
```

### 流式处理

- 处理 `reasoning_content`: 思考过程内容
- 处理 `content`: 最终回答内容
- 按顺序显示，先思考后回答

## 📝 使用示例

配置完成后，运行系统时会看到类似输出：

```
🔧 开始SQL重写...
====================思考过程====================
我需要分析这个SQL查询的结构：
1. 这是一个SELECT语句，包含JOIN和WHERE条件
2. 存在EXISTS子查询，可能可以优化为JOIN
3. 考虑使用FILTER_SUB_QUERY_TO_CORRELATE规则...

====================完整回复====================
重写后的SQL语句如下...
```

## ⚠️ 注意事项

1. **API费用**: 思考模式可能会增加API调用费用
2. **响应时间**: 思考模式会增加响应时间
3. **兼容性**: 只有DeepSeek模型支持此功能，其他模型会使用标准模式

## 🧪 测试

运行测试脚本来验证配置：

```bash
cd /root/AgentRewrite
python3 test_deepseek_thinking.py
```
