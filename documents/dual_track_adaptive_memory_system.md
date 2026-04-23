# 双轨自适应记忆系统技术文档
## Dual-Track Adaptive Memory System Technical Documentation

**版本**: v1.0  
**日期**: 2026-04-17  
**适用系统**: AgentRewrite Query Optimizer  
**文档路径**: `/root/AgentRewrite/documents/dual_track_adaptive_memory_system.md`

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [参数化学习轨迹（Track 1）](#3-参数化学习轨迹track-1)
4. [非参数化记忆轨迹（Track 2）](#4-非参数化记忆轨迹track-2)
5. [双轨协同机制](#5-双轨协同机制)
6. [核心算法与伪代码](#6-核心算法与伪代码)
7. [配置与部署](#7-配置与部署)
8. [实验与评估](#8-实验与评估)
9. [附录](#9-附录)

---

## 1. 系统概述

### 1.1 设计目标

双轨自适应记忆系统旨在解决SQL查询重写框架中的**持续学习**与**经验复用**问题，具体目标包括：

| 目标 | 描述 | 实现方式 |
|------|------|----------|
| **免离线微调** | 无需预训练阶段，部署即运行 | 在线强化学习（LinUCB） |
| **持续进化** | 系统性能随使用次数提升 | 真实执行反馈闭环 |
| **经验复用** | 跨查询共享成功案例 | 向量检索全局记忆库 |
| **低开销** | 毫秒级响应，资源占用可控 | 轻量级向量DB + 矩阵运算 |

### 1.2 核心创新

1. **双轨互补架构**：首次将**参数化在线学习**（LinUCB Bandit）与**非参数化记忆检索**（Vector DB）结合用于SQL重写
2. **真实反馈闭环**：利用数据库执行代价作为奖励信号，摆脱对LLM自我评估的依赖
3. **零人工标注进化**：系统完全自监督运行，无需人工标注优化案例
4. **跨查询经验迁移**：SQL指纹机制实现结构相似查询间的经验复用

### 1.3 系统定位

```
输入SQL → PlanAnalyzer(执行计划分析) → 双轨记忆系统 → Rewrite Agent → 执行验证 → 反馈
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Track 1: LinUCB     Track 2: 向量检索
              规则选择              案例复用
```

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dual-Track Adaptive Memory                  │
├─────────────────────────────┬───────────────────────────────────┤
│   Track 1: Parametric       │   Track 2: Non-Parametric         │
│   (LinUCB Rule Bandit)      │   (Vector Global Memory)          │
├─────────────────────────────┼───────────────────────────────────┤
│ • RuleBandit                │ • SQLFingerprintGenerator         │
│ • Context Extractor (64-dim)│ • VectorStore (ChromaDB)          │
│ • Online Update Module      │ • KnowledgeRetriever              │
│ • UCB Scoring Engine        │ • KnowledgeStorage                │
└─────────────────────────────┴───────────────────────────────────┘
                    │                      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  GlobalMemoryManager │
                    │  (Unified Interface) │
                    └──────────────────────┘
```

### 2.2 模块职责

| 模块 | 文件路径 | 核心职责 |
|------|----------|----------|
| `RuleBandit` | `src/Query_Rewriter/rule_bandit.py` | LinUCB算法实现，规则置信度在线更新 |
| `VectorStore` | `src/Query_Rewriter/global_memory/vector_store.py` | 向量数据库存储与检索（ChromaDB） |
| `KnowledgeRetriever` | `src/Query_Rewriter/global_memory/knowledge_retriever.py` | 相似案例检索与格式化 |
| `KnowledgeStorage` | `src/Query_Rewriter/global_memory/knowledge_storage.py` | 成功案例沉淀与过滤 |
| `SQLFingerprintGenerator` | `src/Query_Rewriter/global_memory/sql_fingerprint.py` | SQL标准化指纹生成 |
| `GlobalMemoryManager` | `src/Query_Rewriter/global_memory/global_memory_manager.py` | 统一接口封装 |

### 2.3 数据流向

```
新SQL查询
    │
    ├──→ [Context Extraction]
    │       ├─ SQL Fingerprint → 32-dim hash
    │       ├─ Plan Bottleneck Operators → 24-dim weighted encoding
    │       └─ Advice Groups → 8-dim one-hot
    │       → 64-dim context vector (L2-normalized)
    │
    ├──→ [Track 1: LinUCB]
    │       ├─ Load rule statistics (A, b matrices)
    │       ├─ Calculate UCB scores: θ^T·x + α·√(x^T·A^(-1)·x)
    │       └─ Return Top-K rules sorted by UCB score
    │
    └──→ [Track 2: Vector Retrieval]
            ├─ Generate SQL fingerprint template
            ├─ Embed with MiniLM-L6-v2 (384-dim)
            ├─ Query ChromaDB (cosine similarity)
            └─ Return Top-K historical cases
    
    ↓
    [Rewrite Agent整合]
    
    执行结果反馈
    │
    ├──→ [Update Track 1]
    │       reward = clip((C_orig - C_rewrite)/C_orig, -1, 1)
    │       A += x·x^T, b += reward·x
    │       Save updated statistics
    │
    └──→ [Update Track 2]
            if cost_reduction > threshold:
                store_successful_case()
```

---

## 3. 参数化学习轨迹（Track 1）

### 3.1 问题建模

将**规则选择**建模为**上下文多臂老虎机（Contextual Multi-Armed Bandit）**问题：

| 组件 | 数学定义 | 实际含义 |
|------|----------|----------|
| **臂（Arm）** | $a \in \mathcal{A}$ | 重写规则（如 `FILTER_INTO_JOIN`） |
| **上下文（Context）** | $x \in \mathbb{R}^{64}$ | SQL特征向量（指纹+算子+组） |
| **奖励（Reward）** | $r \in [-1, 1]$ | 执行成本降低率 |
| **策略（Policy）** | $\pi(a\|x)$ | 选择规则的概率分布 |

### 3.2 上下文特征工程

#### 3.2.1 特征向量构成（64维）

```python
def extract_context(sql: str, explain_info: str, advice_groups: List[str]) -> np.ndarray:
    # 32 dims: SQL指纹的SHA256哈希
    fp = _fingerprint_hash_features(sql)
    
    # 24 dims: 执行计划瓶颈算子（【算子名】+ 代价占比加权）
    plan = _plan_bottleneck_features(explain_info)
    
    # 8 dims: 优化组 one-hot 编码
    advice = [1.0 if g in advice_groups else 0.0 
              for g in ALL_GROUPS]  # 8个优化组
    
    features = concatenate([fp, plan, advice])
    return L2_normalize(features)
```

#### 3.2.2 算子特征编码（关键创新）

```python
def _plan_bottleneck_features(plan_text: str) -> np.ndarray:
    vec = np.zeros(24)
    
    # 从执行计划报告中提取【算子名】
    for match in re.finditer(r"【([^】]+)】", plan_text):
        operator = match.group(1)
        
        # 查找后续500字符内的"代价占比: xx%"
        window = plan_text[match.end():match.end()+500]
        pct_match = re.search(r"代价占比:\s*([\d.]+)\s*%", window)
        weight = float(pct_match.group(1)) / 100.0 if pct_match else 1.0
        
        # SHA256哈希分桶到24维
        digest = hashlib.sha256(operator.lower().encode()).digest()
        i0 = int.from_bytes(digest[0:2], "big") % 24
        i1 = int.from_bytes(digest[2:4], "big") % 24
        
        vec[i0] += 0.5 * weight
        vec[i1] += 0.5 * weight
    
    return np.minimum(vec, 1.0)
```

**关键优势**：
- 无需维护固定算子白名单
- 新算子（如 `Bitmap Heap Scan`, `Memoize`）自动进入特征空间
- 代价占比加权反映算子重要性

### 3.3 LinUCB算法详解

#### 3.3.1 参数估计

对每条规则 $a$，维护：
- $A_a \in \mathbb{R}^{d \times d}$：精度矩阵（协方差逆）
- $b_a \in \mathbb{R}^{d}$：奖励向量

参数估计：
$$\hat{\theta}_a = A_a^{-1} b_a$$

#### 3.3.2 UCB Score计算

$$\text{UCB}(a|x) = \underbrace{\hat{\theta}_a^T x}_{\text{预测奖励}} + \alpha \underbrace{\sqrt{x^T A_a^{-1} x}}_{\text{不确定性奖励}} + \frac{0.1}{1 + n_a}$$

其中：
- $\alpha$：探索参数（默认1.0）
- $n_a$：规则尝试次数（冷启动保护项）

#### 3.3.3 在线更新机制

当观察到奖励 $r$ 时：

$$A_a \leftarrow A_a + x x^T$$

$$b_a \leftarrow b_a + r \cdot x$$

**时间复杂度**：$O(d^2)$，其中 $d=64$

### 3.4 序列级奖励分配（LLM效果分数与置信度混合）

#### 3.4.1 核心机制

权重来源由 `DecisionAgent`（推理Agent）在规则选择时输出：

```json
{
    "applied_rules": ["RULE_ID_1", "RULE_ID_2", ...],
    "rule_effect_scores": {"RULE_ID_1": 0.6, "RULE_ID_2": 0.4, ...},
    "rule_effect_confidence": {"RULE_ID_1": 0.8, "RULE_ID_2": 0.7, ...}
}
```

#### 3.4.2 混合权重计算公式

```python
def _mix_llm_effect_with_confidence_gate(
    self,
    applied: List[str],
    effect_scores: Dict[str, float],
    effect_confidence: Dict[str, float],
) -> Dict[str, float]:
    """
    Mixed weight per rule:
        w_i = lambda_i * w_i_llm + (1-lambda_i) * w_i_fallback
        lambda_i = lambda_base * confidence_i
    """
    fallback = self._fallback_position_weights(applied, position_decay=0.90)
    
    # LLM效果分数归一化
    llm_sum = sum(effect_scores.values())
    w_i_llm = {rid: score / llm_sum for rid, score in effect_scores.items()}
    
    # 置信度调节的混合权重
    mixed = {}
    for rid in applied:
        conf = effect_confidence.get(rid, 0.5)
        lambda_i = self.EFFECT_SCORE_LAMBDA_BASE * conf  # lambda_base = 0.5
        mixed[rid] = lambda_i * w_i_llm[rid] + (1.0 - lambda_i) * fallback[rid]
    
    return normalize(mixed)
```

#### 3.4.3 分配策略

| 场景 | 权重来源 | 说明 |
|------|----------|------|
| **高置信度** (conf ≈ 1.0) | 主要依赖LLM效果分数 | `lambda ≈ 0.5`，LLM判断占主导 |
| **低置信度** (conf ≈ 0.0) | 主要依赖几何衰减 | `lambda ≈ 0`，位置权重占主导 |
| **无LLM输出** | 完全回退几何衰减 | 纯位置衰减 `w_i = decay^i` |

**奖励计算公式**：
$$r = \text{clip}\left(\frac{C_{orig} - C_{rewrite}}{C_{orig}}, -1, 1\right)$$

**最终奖励分配**：
$$r_i = r \cdot w_i^{mixed}$$

---

## 4. 非参数化记忆轨迹（Track 2）

### 4.1 系统组成

| 组件 | 技术选型 | 功能描述 |
|------|----------|----------|
| `SQLFingerprintGenerator` | 正则 + sqlparse | SQL标准化与参数化 |
| `VectorStore` | ChromaDB + MiniLM-L6-v2 | 向量存储与相似度检索 |
| `KnowledgeRetriever` | Cosine Similarity | 相似案例检索 |
| `KnowledgeStorage` | 成本降低过滤 | 成功案例沉淀 |

### 4.2 SQL指纹生成

#### 4.2.1 标准化流程

```python
def get_template(sql: str) -> str:
    # Step 1: 移除注释
    sql = remove_comments(sql)
    
    # Step 2: 常量参数化
    sql = re.sub(r'\b\d+\.?\d*\b', '?', sql)      # 数字→?
    sql = re.sub(r"'[^']*'", '?', sql)           # 字符串→?
    
    # Step 3: 大小写规范化
    sql = normalize_case(sql)  # 关键字大写, 标识符小写
    
    # Step 4: 空白规范化
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql
```

#### 4.2.2 示例

| 原始SQL | 指纹模板 |
|---------|----------|
| `SELECT * FROM orders WHERE customer_id = 12345` | `SELECT * FROM orders WHERE customer_id = ?` |
| `SELECT name FROM users WHERE age > 25 AND status = 'active'` | `SELECT name FROM users WHERE age = ? AND status = ?` |

### 4.3 向量存储结构

```python
def add(self, sql_fingerprint, rule_sequence, groups, cost_reduction_rate, ...):
    # 使用MiniLM-L6-v2生成384维嵌入
    embedding = self.embedding_model.encode(sql_fingerprint)
    
    metadata = {
        "sql_fingerprint": fingerprint,
        "rule_sequence": json.dumps(rule_sequence),  # 成功规则序列
        "groups": groups,
        "cost_reduction_rate": str(cost_reduction_rate),
        "frequency": "1"
    }
    
    collection.add(ids=[uuid], embeddings=[embedding], 
                   metadatas=[metadata], documents=[fingerprint])
```

**存储路径**：`data/global_memory/chroma_db/`

### 4.4 成功案例检索

```python
def retrieve(self, sql: str, top_k: int = 3) -> List[Dict]:
    # 生成查询指纹
    fingerprint = self.fingerprint_generator.get_template(sql)
    
    # 向量相似度检索
    results = self.vector_store.query(fingerprint, top_k)
    
    return [{
        "rule_sequence": metadata['rule_sequence'],
        "cost_reduction_rate": float(metadata['cost_reduction_rate']),
        "frequency": int(metadata['frequency']),
        "score": result['score']  # cosine similarity
    } for r in results]
```

### 4.5 成功判定与过滤

```python
def store_successful_optimization(self, original_sql, rewritten_sql, 
                                  rule_sequence, groups, 
                                  original_cost, rewritten_cost):
    # 严格过滤条件
    if original_cost <= 0:
        return None
    
    if rewritten_cost >= original_cost:
        return None  # 仅存储成本降低的案例
    
    cost_reduction_rate = (original_cost - rewritten_cost) / original_cost
    
    # 存储到向量数据库
    return self.vector_store.add(...)
```

---

## 5. 双轨协同机制

### 5.1 协同工作流

```
┌─────────────────────────────────────────────────────────┐
│                    新SQL查询进入                          │
└─────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐ ┌──────────────┐ ┌──────────────┐
    │ 执行计划    │ │ Track 1      │ │ Track 2      │
    │ 分析       │ │ LinUCB       │ │ Vector       │
    └────────────┘ └──────────────┘ └──────────────┘
           │               │               │
           │               ▼               ▼
           │        ┌──────────┐   ┌──────────┐
           │        │ UCB Score │   │ 相似案例  │
           │        │ 排序      │   │ 检索      │
           │        └──────────┘   └──────────┘
           │               │               │
           └───────────────┴───────────────┘
                           ▼
            ┌──────────────────────────────┐
            │     LLM Rewrite Agent        │
            │  - System Prompt: Top-K规则   │
            │  - Context: Few-shot历史案例  │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │       执行重写SQL            │
            │    获取真实执行代价           │
            └──────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │ 反馈Track 1   │              │ 反馈Track 2   │
    │ 更新Bandit   │              │ 沉淀成功案例  │
    │ 参数         │              │              │
    └──────────────┘              └──────────────┘
```

### 5.2 优势互补矩阵

| 维度 | Track 1: 参数化 | Track 2: 非参数化 |
|------|-----------------|-------------------|
| **学习机制** | 在线强化学习 | 基于相似度的案例复用 |
| **存储形式** | 低维参数（A, b矩阵） | 高维向量（384维嵌入） |
| **泛化能力** | 跨查询类型泛化 | 同类型查询精确匹配 |
| **冷启动** | 需要多轮探索 | 立即检索相似案例 |
| **增量更新** | 参数平滑更新 | 直接追加新案例 |
| **可解释性** | UCB分数可解释 | 相似度+成本降低率可解释 |
| **计算开销** | $O(d^2)$ | $O(n \cdot d)$ |

### 5.3 初始化与集成代码

```python
# src/Query_Rewriter/langgraph_rewriter.py
class LangGraphQueryRewriter:
    def __init__(self, ...):
        # Track 1: 参数化学习（LinUCB）
        self.bandit = get_rule_bandit()
        
        # Track 2: 非参数化记忆
        self.global_memory = GlobalMemoryManager()
    
    async def _rule_selection_node(self, state: RewriteState):
        # 提取上下文特征
        context = self.bandit.extract_context(
            state["initial_sql"],
            state["initial_explain_info"],
            groups
        )
        
        # Track 1: 计算所有规则的UCB分数
        scored_rules = self.bandit.score_rules(rule_library, context)
        
        # Track 2: 检索相似案例作为Few-shot
        if self.global_memory:
            similar_cases = self.global_memory.retrieve(
                state["initial_sql"], top_k=3
            )
            few_shot = self._format_similar_cases(similar_cases)
        
        # 整合双轨信息，调用LLM
        selected_rules = await self.decision_agent.select_rules(
            context=context,
            candidate_rules=scored_rules[:10],
            few_shot_examples=few_shot
        )
        
        return {"selected_rules": selected_rules, ...}
    
    async def _evaluation_node(self, state: RewriteState):
        # ... 评估重写效果 ...
        
        # Track 1: 更新Bandit参数
        sequence_reward = self.bandit.compute_sequence_reward(
            original_cost, rewritten_cost
        )
        self.bandit.update_with_sequence_reward(
            applied_rules, context, sequence_reward
        )
        
        # Track 2: 沉淀成功案例
        if improvement and self.global_memory:
            self.global_memory.store_successful_optimization(
                original_sql=state["initial_sql"],
                rewritten_sql=best_sql,
                rule_sequence=applied_rules,
                groups=groups,
                original_cost=original_cost,
                rewritten_cost=rewritten_cost
            )
```

---

## 6. 核心算法与伪代码

### 6.1 Algorithm 1: LinUCB规则选择

```
Algorithm: LinUCB-based Rule Selection
Input: 规则库 A, 上下文 x ∈ ℝ^64, 探索参数 α
Output: 选择的规则 a*

for each rule a ∈ A do
    θ̂_a ← A_a^(-1) b_a                    # 参数估计
    μ_a ← θ̂_a^T x                        # 预测奖励
    σ_a ← √(x^T A_a^(-1) x)              # 不确定性
    UCB_a ← μ_a + α · σ_a + 0.1/(1+n_a)
end for

a* ← argmax_{a ∈ A} UCB_a
return a*
```

### 6.2 Algorithm 2: 上下文特征提取

```
Algorithm: Context Feature Extraction
Input: SQL查询 q, 执行计划分析 p, 优化组列表 G
Output: 64维上下文向量 x

// 32-dim: SQL指纹哈希
tpl ← FingerprintGenerator(q)
h ← SHA256(tpl)
x_fp ← Normalize(BytesToFloat(h))[:32]

// 24-dim: 算子瓶颈特征
x_plan ← 0_24
for each operator op in p with pattern 【op】 do
    w ← ExtractCostPercentage(p, op)
    digest ← SHA256(op)
    i0 ← BytesToInt(digest[0:2]) mod 24
    i1 ← BytesToInt(digest[2:4]) mod 24
    x_plan[i0] += 0.5 · w
    x_plan[i1] += 0.5 · w
end for
x_plan ← min(x_plan, 1.0)

// 8-dim: 优化组 one-hot
all_groups ← [子查询, 连接, 谓词, 常量, 聚合, 投影, 排序, 集合]
x_advice ← [𝟙(g ∈ G) for g in all_groups]

x ← Concat(x_fp, x_plan, x_advice)
x ← x / ||x||_2                         # L2归一化
return x
```

### 6.3 Algorithm 3: 序列奖励分配（LLM效果分数与置信度混合）

```
Algorithm: Mixed Reward Allocation with Confidence Gate
Input: 应用规则序列 rules, LLM效果分数 scores, LLM置信度 confidences,
       序列级奖励 R, 位置衰减系数 gamma=0.90, lambda_base=0.5
Output: 各规则分配奖励 {rule_id: reward_i}

// 步骤1: 计算Fallback几何衰减权重
Function FallbackPositionWeights(rules, gamma):
    n ← length(rules)
    for i ← 0 to n-1 do
        w_fallback[rules[i]] ← gamma^i
    end for
    return Normalize(w_fallback)
End Function

// 步骤2: 计算LLM效果权重
w_fallback ← FallbackPositionWeights(rules, gamma)

if scores is empty then
    return {rid: R * w_fallback[rid] for rid in rules}
end if

w_llm ← Normalize(scores)  // 归一化效果分数

// 步骤3: 置信度门控混合
mixed ← {}
for rid in rules do
    lambda_i ← lambda_base * confidences.get(rid, 0.5)
    mixed[rid] ← lambda_i * w_llm[rid] + (1 - lambda_i) * w_fallback[rid]
end for

w_final ← Normalize(mixed)

// 步骤4: 分配奖励
allocated ← {}
for rid in rules do
    allocated[rid] ← R * w_final[rid]
end for

return allocated
```

### 6.4 Algorithm 4: 成功案例检索与存储

```
Algorithm: Vector-based Knowledge Retrieval and Storage
Input: 查询SQL q, 向量数据库 D, 检索数量 k
Output: 相似案例列表 C

// 检索阶段
tpl ← GenerateFingerprint(q)
e ← MiniLM(tpl)                          # 384-dim embedding
C ← TopK(D, e, k, cosine)

// 存储阶段（成功后）
Function StoreSuccessfulCase(q, q', rules, C_orig, C_new):
    if C_new ≥ C_orig or C_orig ≤ 0 then
        return null                       # 仅存储成本降低案例
    end if
    rate ← (C_orig - C_new) / C_orig
    tpl ← GenerateFingerprint(q)
    e ← MiniLM(tpl)
    D.add(e, metadata={rules, rate, ...})
End Function
```

### 6.5 核心公式汇总

**UCB Score**：
$$\text{UCB}(a|x) = \hat{\theta}_a^T x + \alpha \sqrt{x^T A_a^{-1} x} + \frac{0.1}{1+n_a}$$

**参数更新**：
$$A_a \leftarrow A_a + x x^T, \quad b_a \leftarrow b_a + r \cdot x$$

**奖励计算**：
$$r = \text{clip}\left(\frac{C_{orig} - C_{rewrite}}{C_{orig}}, -1, 1\right)$$

**序列奖励分配**（LLM效果分数与置信度混合）：

$$w_i^{LLM} = \frac{\text{effect\_score}_i}{\sum_j \text{effect\_score}_j}$$

$$\lambda_i = \lambda_{base} \cdot \text{confidence}_i$$

$$w_i^{mixed} = \lambda_i \cdot w_i^{LLM} + (1 - \lambda_i) \cdot \frac{\gamma^i}{\sum_j \gamma^j}$$

$$r_i = r \cdot w_i^{mixed}$$

---

## 7. 配置与部署

### 7.1 环境依赖

```bash
# requirements.txt
numpy>=1.21.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
sqlparse>=0.4.0
```

### 7.2 配置参数

| 参数 | 默认值 | 说明 | 配置位置 |
|------|--------|------|----------|
| `context_dim` | 64 | 上下文特征维度 | `rule_bandit.py` |
| `alpha` | 1.0 | LinUCB探索参数 | `RuleBandit.__init__` |
| `position_decay` | 0.90 | 序列奖励衰减率 | `update_with_sequence_reward` |
| `top_k` | 3 | 向量检索案例数 | `KnowledgeRetriever.retrieve` |
| `embedding_model` | `all-MiniLM-L6-v2` | 嵌入模型 | 环境变量 |

### 7.3 存储路径

```
data/
└── global_memory/
    ├── chroma_db/                    # ChromaDB向量存储
    │   ├── chroma.sqlite3
    │   └── ...
    └── rule_bandit_stats.json        # LinUCB参数持久化
```

### 7.4 单机/分布式部署

**单机模式**：
```python
from src.Query_Rewriter.global_memory import GlobalMemoryManager
from src.Query_Rewriter.rule_bandit import get_rule_bandit

# 自动初始化，数据持久化到本地文件
bandit = get_rule_bandit()
memory = GlobalMemoryManager()
```

**分布式注意事项**：
- `rule_bandit_stats.json` 需使用共享存储（NFS/S3）
- ChromaDB支持服务端模式部署

---

## 8. 实验与评估

### 8.1 对比实验设计

| 实验组 | 配置 | 目的 |
|--------|------|------|
| **Baseline** | 无记忆系统，纯LLM重写 | 验证基础性能 |
| **Track 1 Only** | 仅LinUCB，无向量检索 | 验证参数化学习效果 |
| **Track 2 Only** | | 仅向量检索，无LinUCB | 验证非参数化记忆效果 |
| **Full System** | 双轨完整系统 | 验证协同优势 |
| **Ablation** | 不同$\alpha$值（0.5, 1.0, 2.0） | 探索-利用权衡 |

### 8.2 评估指标

| 指标 | 定义 | 测量方式 |
|------|------|----------|
| **成本降低率** | $\frac{C_{orig} - C_{best}}{C_{orig}}$ | EXPLAIN代价 |
| **成功率** | 成本降低>10%的查询占比 | 统计计数 |
| **收敛速度** | 达到稳定性能所需查询数 | 滑动窗口 |
| **冷启动表现** | 前N个查询的平均性能 | 分段统计 |
| **存储开销** | 内存/磁盘占用 | 系统监控 |

### 8.3 预期结果

| 场景 | 预期表现 |
|------|----------|
| **冷启动** | Track 2提供即时案例复用，Track 1快速探索 |
| **稳定期** | 双轨协同，成本降低率持续提升 |
| **长尾查询** | Track 1泛化能力强，Track 2积累特异性经验 |
| **概念漂移** | 在线更新自适应，无需重新训练 |

### 8.4 可视化建议

1. **学习曲线**：横轴查询序号，纵轴平均成本降低率
2. **UCB分数分布**：不同规则的UCB分数随时间变化
3. **案例命中率**：向量检索成功案例的比例
4. **特征重要性**：64维特征对规则选择的影响

---

## 9. 附录

### A. 数学推导

**LinUCB regret bound**（标准结果）：

$$\text{Regret}(T) \leq O\left(d\sqrt{T} \log T\right)$$

其中 $d=64$ 为上下文维度。

### B. 接口定义

```python
class RuleBandit:
    def extract_context(self, sql: str, explain_info: str, 
                        advice_groups: List[str]) -> np.ndarray: ...
    def calculate_ucb_score(self, rule_id: str, context: np.ndarray) -> float: ...
    def update_with_sequence_reward(self, rule_sequence: List[str], 
                                     context: np.ndarray, 
                                     sequence_reward: float) -> Dict[str, float]: ...

class GlobalMemoryManager:
    def retrieve(self, sql: str, top_k: int = 3) -> List[Dict]: ...
    def store_successful_optimization(self, original_sql: str, 
                                       rewritten_sql: str,
                                       rule_sequence: List[str],
                                       original_cost: float,
                                       rewritten_cost: float) -> Optional[str]: ...
```


---

**文档结束**

*本系统通过参数化与非参数化两条轨迹，赋予了框架免除离线微调开销即可实现长效进化的能力，大幅降低了LLM的试错成本并提升了系统的在线演进效率。*
