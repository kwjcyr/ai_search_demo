from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------
# 0. 模拟内容库 (外挂知识库 - 包含冷启动内容)
# ---------------------------------------------------------
item_pool = [
    "深度学习入门：从神经元到Transformer",
    "2025年全球宏观经济展望",
    "极简主义生活指南：如何通过断舍离获得快乐",
    "Python异步编程实战：Asyncio完全解析",
    "法式甜点制作入门：马卡龙的秘密",
    "硅谷最新动态：OpenAI算力之争",
    "室内绿植养护手册：让你的客厅变成森林",
    "Rust语言为什么是系统编程的未来？",
    "徒步爱好者天堂：尼泊尔攻略",
    "高效能人士的7个习惯"
]

# 初始化 Embedding 模型 (使用 BGE 中文增强版)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 构建向量库
vector_db = FAISS.from_texts(item_pool, embeddings)
# 初始化本地大模型
llm = Ollama(model="gemma2:2b")

# ---------------------------------------------------------
# Step 1: 行为序列 Listwise 重排 (Reranking Click History)
# 核心：从一串点击中，通过逻辑推理找出“谁才是用户现在的真爱”
# ---------------------------------------------------------
def listwise_history_rerank(click_history):
    print(f"\n[Step 1] 原始点击序列: {click_history}")
    prompt = PromptTemplate.from_template("""
    用户最近点击了以下内容（按时间顺序）：
    {history}

    任务：分析这些点击，判断哪些是“误点/噪音”，哪些是“核心兴趣”。
    请按“兴趣强度”从高到低对这几个标题重新排序。
    注意：直接输出标题，不要包含序号或解释，使用中文。
    排序结果：""")

    response = llm.invoke(prompt.format(history="\n".join(click_history))).strip()
    # 取重排后的第一个作为核心兴趣
    core_interest = response.split('\n')[0].strip()
    print(f"🎯 Listwise 重排结果（核心兴趣锁定）: {core_interest}")
    return core_interest

# ---------------------------------------------------------
# Step 2: Zero-shot 意图提取 (Intent Extraction)
# 核心：基于核心兴趣，跨越标签，直接进行语义推理
# ---------------------------------------------------------
def get_zero_shot_intent(core_item):
    print(f"\n[Step 2] 正在进行 Zero-shot 意图推理...")
    prompt = PromptTemplate.from_template("""
    用户当前最核心的兴趣点是：'{item}'

    请推测用户当前的深层需求是什么？他想学习什么技能或解决什么问题？
    请用中文简短描述。
    意图描述：""")

    intent = llm.invoke(prompt.format(item=core_item)).strip()
    print(f"🤖 推理出的意图：{intent}")
    return intent

# ---------------------------------------------------------
# Step 3: 语义召回 (Cold-start Recall)
# 核心：解决冷启动，新内容靠语义被捞出
# ---------------------------------------------------------
def cold_start_recall(intent):
    print(f"\n[Step 3] 正在从外挂知识库召回（解决冷启动）...")
    # 拿着 LLM 生成的意图去向量库里“捞”
    docs = vector_db.similarity_search(intent, k=3)
    candidates = [d.page_content for d in docs]
    print(f"🔍 召回结果: {candidates}")
    return candidates

# ---------------------------------------------------------
# Step 4: 解释性呈现 (Generative Presentation)
# ---------------------------------------------------------
def final_presentation(item, intent):
    print(f"\n[Step 4] 正在生成推荐语...")
    prompt = PromptTemplate.from_template("""
    推荐内容：{item}
    用户意图：{intent}
    任务：请写一句吸引人的中文推荐语，解释为什么这个内容符合用户的兴趣。
    要求：
    1. 必须完全使用中文。
    2. 不要包含任何英文解释或说明。
    3. 语气要亲切、专业。
    推荐语：""")
    return llm.invoke(prompt.format(item=item, intent=intent)).strip()

if __name__ == "__main__":
    # 模拟用户行为：前两个是误点，最后一个是深度阅读
    user_clicks = [
        "法式甜点制作入门",
        "室内绿植养护手册",
        "大模型时代的程序员生存指南"
    ]

    print("="*80)
    print("🚀 AI 推荐系统工作流启动 (中文版)")
    print("="*80)

    # 1. Listwise 重排历史：从点击序列中找到“真爱”
    core_click = listwise_history_rerank(user_clicks)

    # 2. Zero-shot 提取意图：理解“真爱”背后的逻辑
    intent = get_zero_shot_intent(core_click)

    # 3. 召回：寻找库里最匹配的新内容（冷启动）
    candidates = cold_start_recall(intent)

    # 4. 呈现：生成推荐语
    if candidates:
        top_item = candidates[0]
        rec_msg = final_presentation(top_item, intent)

        print("\n" + "✨" * 30)
        print(f"【最终推荐】: {top_item}")
        print(f"【推荐理由】: {rec_msg}")
        print("✨" * 30)

    print("\n" + "="*80)

