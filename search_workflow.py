import os

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------
# 1. 构造一批“文档”
# ----------------------
docs = [
    "用户下单流程：Controller → OrderService → OrderManager → OrderDao → DB",
    "OOM 通常是 batch size 太大、模型加载太多、内存泄漏导致",
    "Java 虚拟机栈溢出一般是递归深度太深或循环调用",
    "AI Search 包含：意图理解、召回、精排、生成",
    "RLHF 是用强化学习让模型更符合人类意图",
    "记忆系统是因为 LLM 上下文不够才出现的补丁方案",
]

# ----------------------
# 2. 向量库持久化逻辑
# ----------------------
DB_PATH = "faiss_index_v2"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

def get_vector_db_and_splits():
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = splitter.create_documents(docs)

    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(splits, embeddings)
        db.save_local(DB_PATH)
    return db, splits

db, splits = get_vector_db_and_splits()

# ----------------------
# 3. 混合检索器 (Hybrid Search)
# ----------------------
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4
faiss_retriever = db.as_retriever(search_kwargs={"k": 4})
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

# ----------------------
# 4. 细粒度步骤定义
# ----------------------
llm = Ollama(model="gemma2:2b")

# Step 1: 意图识别
def get_intent(query):
    prompt = PromptTemplate.from_template("分析用户问题，输出唯一意图（code_debug, architecture, tech_concept, unknown）。问题：{query}\n意图：")
    return llm.invoke(prompt.format(query=query)).strip()

# Step 2: 关键词提取 (用于增强检索)
def get_keywords(query):
    prompt = PromptTemplate.from_template("从用户问题中提取 2-3 个核心关键词，用逗号隔开。问题：{query}\n关键词：")
    keywords = llm.invoke(prompt.format(query=query)).strip()
    print(f"🔑 提取关键词：{keywords}")
    return keywords

# Step 3: 结果校验 (质检员)
def verify_answer(query, context, answer):
    prompt = PromptTemplate.from_template("""
    你是一个严谨的质检员。请判断给出的回答是否完全基于参考信息。
    如果回答中包含参考信息里没有的内容，请输出 'FAIL' 并说明原因。
    如果回答正确，请输出 'PASS'。

    参考信息：{context}
    用户问题：{query}
    给出回答：{answer}

    判断结果：""")
    verification = llm.invoke(prompt.format(query=query, context=context, answer=answer)).strip()
    print(f"✅ 质检结果：{verification}")
    return "PASS" in verification.upper()

# ----------------------
# 5. AI Search：Agentic Workflow (工作流模式)
# ----------------------
def ai_search_workflow(query):
    print(f"\n🚀 开始处理问题：{query}")

    # 1. 意图识别
    intent = get_intent(query)
    print(f"🤖 识别意图：{intent}")

    # 2. 关键词提取并检索
    keywords = get_keywords(query)
    # 使用关键词进行检索，通常比原始长句更准
    retrieved_docs = ensemble_retriever.invoke(keywords)
    context = "\n".join([d.page_content for d in retrieved_docs[:2]])
    print(f"🔍 检索到 {len(retrieved_docs)} 条相关信息")

    # 3. 生成初步回答
    answer_prompt = PromptTemplate.from_template("用户意图：{intent}\n参考信息：\n{context}\n\n问题：{query}\n请简洁回答。")
    initial_answer = llm.invoke(answer_prompt.format(intent=intent, context=context, query=query))

    # 4. 结果校验
    is_valid = verify_answer(query, context, initial_answer)

    if is_valid:
        return initial_answer
    else:
        return f"⚠️ 警告：模型生成可能存在幻觉，请谨慎参考。\n初步回答：{initial_answer}"

if __name__ == "__main__":
    q = "OOM 怎么产生的？"
    print("="*80)
    final_res = ai_search_workflow(q)
    print(f"\n🏁 最终回答：\n{final_res}")
    print("="*80)
