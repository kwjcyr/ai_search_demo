# AI Search & Recommendation Demo (LLM-based Architecture)

这是一个展示主流 AI 搜索（RAG）与推荐系统架构的 Demo 项目。它体现了“大脑（LLM）+ 外挂（知识库/行为序列）”的核心思想，并展示了如何通过细粒度的工作流（Workflow）提升小模型的表现。

## 🚀 核心理念
*   **大脑与外挂分离**：大模型（如 Qwen, Gemma）作为逻辑推理引擎，其权重是静态的；而知识库（向量库）和用户行为序列作为“外挂记忆”，是动态更新的。
*   **流程胜过模型**：通过将任务拆解为意图识别、关键词提取、混合检索、Listwise 重排和结果校验等细粒度步骤，使 2B/8B 的小模型也能达到甚至超过超大模型的单次调用效果。

## 📂 文件说明

### 1. `search_workflow.py` (AI 搜索/RAG 工作流)
该脚本演示了一个完整的本地 AI 搜索流程：
*   **意图识别**：分析用户问题，判断是查代码、查架构还是查概念。
*   **关键词提取**：自动从长句中提取核心词，优化检索精度。
*   **混合检索 (Hybrid Search)**：结合了 **BM25 (关键词匹配)** 和 **FAISS (语义向量检索)**，并使用 **RRF (倒数排名融合)** 算法解决向量检索容易忽略硬核关键词的问题。
*   **结果校验 (Self-Correction)**：引入“质检员”步骤，AI 会自我检查回答是否完全基于参考文档，有效防止幻觉。

### 2. `rec_workflow.py` (AI 推荐工作流)
该脚本演示了基于行为序列的现代推荐逻辑：
*   **Listwise 行为重排**：从一串点击历史中，通过逻辑推理找出用户的“核心兴趣点”，剔除误点噪音。
*   **Zero-shot 意图提取**：将行为序列翻译成自然语言描述的“深层需求”，无需预定义标签。
*   **冷启动召回**：利用语义向量检索，即使是点击量为 0 的新内容，只要语义契合也能被精准分发。
*   **解释性呈现**：不仅推给你内容，还告诉你“为什么推给你”。

## 🛠️ 运行环境

### 1. 安装依赖
```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers ollama rank_bm25
```

### 2. 启动本地 LLM (Ollama)
本项目默认使用 `gemma2:2b` 模型，请确保已安装并启动 Ollama：
```bash
# 启动服务
ollama serve

# 下载模型
ollama pull gemma2:2b
```

### 3. 运行 Demo
```bash
# 运行 AI 搜索 Demo
python3 search_workflow.py

# 运行 AI 推荐 Demo
python3 rec_workflow.py
```

## 🔗 关于作者
*   GitHub: [github.com/kwjcyr/ai_search_demo](https://github.com/kwjcyr/ai_search_demo)

