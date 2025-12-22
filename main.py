import os
import json
import datetime
import requests
import smtplib
import akshare as ak
import arxiv
from email.mime.text import MIMEText
from email.header import Header
from openai import OpenAI
from serpapi import GoogleSearch

# ==========================================
#              1. 全局配置区域 (CONFIG)
#           所有你想改的参数都在这里！
# ==========================================

CONFIG = {
    # --- 基础设置 ---
    "DATA_FILE": "data/reports.json", # 数据存储路径
    "MAX_HISTORY": 500,               # 历史记录保留多少条
    "MIN_SCORE": 5.0,                 # 低于这个分的文章不收录
    "PUSH_THRESHOLD": 6,            # 高于这个分才推送到钉钉/手机
    
    # --- 抓取数量 (每个源尝试获取的原始条数) ---
    "FETCH_COUNT_ARXIV": 20,          # ArXiv 抓多点，因为要过滤
    "FETCH_COUNT_GOOGLE": 10,
    "FETCH_COUNT_AKSHARE": 50,        # A股研报杂音多，抓50条回来筛选
    "FINAL_SAVE_COUNT": 15,           # 最终每天保存并展示的最大条数
    
    # --- ArXiv (国际论文) 设置 ---
    # 搜索关键词 (逻辑是 OR)
    "ARXIV_KEYWORDS": [
        "quantitative finance",
        "factor model",
        "portfolio optimization",
        "deep learning trading",
        "market microstructure",
        "risk premia",
        "machine learning trading",
        "reinforcement learning trading",
        "algorithm trading"
    ],
    
    # --- Google Scholar (谷歌学术) 设置 ---
    "GOOGLE_QUERY": 'quantitative finance "machine learning" trading after:2024',
    
    # --- Akshare (国内研报) 设置 ---
    # 1. 必读券商白名单 (只看这些金工强队的报告)
    "TARGET_BROKERS": [
        "中信建投", "华泰证券", "天风证券", "兴业证券", 
        "国泰君安", "招商证券", "中金公司", "申万宏源",
        "海通证券", "广发证券"
    ],
    # 2. 标题关键词 (必须包含其中之一)
    "AK_KEYWORDS": [
        "金工", "量化", "因子", "选股", "择时", 
        "资产配置", "深度研究", "基本面量化", "多因子",
        "机器学习", "神经网络", "高频"
    ]
}

# ==========================================
#              2. 环境变量加载
# ==========================================
LLM_API_KEY = os.environ.get("LLM_API_KEY")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

# 初始化 AI 客户端
client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
#              3. 核心抓取逻辑
# ==========================================

def fetch_arxiv():
    """抓取 ArXiv"""
    print(f"--- 正在抓取 ArXiv (关键词: {CONFIG['ARXIV_KEYWORDS'][:3]}...) ---")
    
    # 构造查询语句: cat:q-fin.* AND ("keyword1" OR "keyword2" ...)
    keywords_query = " OR ".join([f'"{k}"' for k in CONFIG['ARXIV_KEYWORDS']])
    query = f'cat:q-fin.* AND ({keywords_query})'
    
    search = arxiv.Search(
        query=query,
        max_results=CONFIG['FETCH_COUNT_ARXIV'],
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    for r in search.results():
        results.append({
            "title": r.title,
            "url": r.pdf_url,
            "source": "ArXiv",
            "date": r.published.strftime("%Y-%m-%d"),
            "abstract": r.summary,
            "broker": "Cornell Univ" 
        })
    return results

def fetch_google_scholar():
    """抓取 Google Scholar"""
    if not SERPAPI_KEY:
        print("未配置 SERPAPI_KEY，跳过 Google Scholar")
        return []
        
    print(f"--- 正在抓取 Google Scholar ---")
    params = {
        "engine": "google_scholar",
        "q": CONFIG['GOOGLE_QUERY'],
        "api_key": SERPAPI_KEY,
        "num": CONFIG['FETCH_COUNT_GOOGLE']
    }
    search = GoogleSearch(params)
    data = search.get_dict()
    organic_results = data.get("organic_results", [])
    
    results = []
    for item in organic_results:
        results.append({
            "title": item.get("title"),
            "url": item.get("link"),
            "source": "Scholar",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"), # 谷歌学术很难获取精确日期，用当天代替
            "abstract": item.get("snippet", "No abstract available"),
            "broker": "Google"
        })
    return results

def fetch_akshare_reports():
    """抓取 A股金工研报 (核心逻辑增强版)"""
    print(f"--- 正在抓取 A股研报 (目标: {len(CONFIG['TARGET_BROKERS'])}家券商) ---")
    results = []
    try:
        # 获取最近的研报数据 (默认取最新的 100 条 raw data 来筛选)
        # 注意：akshare 接口返回的是全市场的，我们需要在内存里做筛选
        target_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # 为了防止周末没有研报，如果今天没有，可以尝试往前推（这里简化，只抓当天的接口数据）
        # stock_em_yjbg 接口返回的是最近一个交易日更新的列表
        df = ak.stock_em_yjbg(date=target_date)
        
        if df.empty:
            print("今日 Akshare 接口暂无数据")
            return []

        # === 筛选逻辑 1: 必须是白名单券商 ===
        # 假设 df 里的列名是 '机构名称'
        df = df[df['机构名称'].isin(CONFIG['TARGET_BROKERS'])]
        
        # === 筛选逻辑 2: 标题必须包含关键词 ===
        # 使用正则表达式构建 "A|B|C"
        keywords_pattern = "|".join(CONFIG['AK_KEYWORDS'])
        df = df[df['文章标题'].str.contains(keywords_pattern, na=False)]
        
        # 取前 N 条
        df = df.head(CONFIG['FETCH_COUNT_AKSHARE'])
        
        for _, row in df.iterrows():
            results.append({
                "title": row['文章标题'],
                "url": row['pdf链接'], 
                "source": "研报",
                "date": row['发布日期'],
                "abstract": f"来自 {row['机构名称']} 的深度报告：{row['文章标题']}", # 研报无摘要，用这个代替
                "broker": row['机构名称']
            })
            
    except Exception as e:
        print(f"Akshare 抓取异常 (可能是接口变动或网络问题): {e}")
        
    return results

# ==========================================
#              4. 智能分析与分发
# ==========================================

def analyze_with_llm(item):
    """调用 AI 进行评分和总结"""
    try:
        # 针对研报和论文使用不同的 Prompt 策略
        content_type = "学术论文" if item['source'] in ['ArXiv', 'Scholar'] else "A股金工研报"
        
        prompt = f"""
        你是一个专业的量化基金经理。请评估以下{content_type}的价值。
        
        标题: {item['title']}
        来源: {item['broker']}
        摘要/内容: {item['abstract'][:800]}
        
        请严格按 JSON 格式返回：
        {{
            "score": <0-10分, 7分代表有实战参考价值, 9分代表必读>,
            "summary": "<用中文一句话概括核心策略或创新点,不超过50字>"
        }}
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM 分析失败: {e}")
        return {"score": 5.0, "summary": "AI 分析暂时不可用"}

def send_dingtalk(msg_markdown):
    """发送钉钉消息"""
    if not DINGTALK_WEBHOOK: return
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": "量化日报推送",
                "text": msg_markdown
            }
        }
        requests.post(DINGTALK_WEBHOOK, json=data)
    except Exception as e:
        print(f"钉钉发送失败: {e}")

def send_email(subject, html_content):
    """发送邮件"""
    if not EMAIL_USER or not EMAIL_PASS: return
    try:
        msg = MIMEText(html_content, 'html', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_USER 
        
        smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)
        smtp.quit()
    except Exception as e:
        print(f"邮件发送失败: {e}")

# ==========================================
#              5. 主程序入口
# ==========================================

def main():
    print(">>> 任务开始")
    
    # 1. 加载历史数据 (去重用)
    history_ids = []
    if os.path.exists(CONFIG["DATA_FILE"]):
        with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
            old_data = json.load(f)
            history_ids = [item.get('title') for item in old_data] # 用标题做唯一ID

    # 2. 抓取所有源
    raw_items = []
    raw_items += fetch_arxiv()
    raw_items += fetch_google_scholar()
    raw_items += fetch_akshare_reports()
    
    print(f">>> 共抓取到 {len(raw_items)} 条原始数据，开始 AI 筛选...")

    # 3. AI 分析与筛选
    new_qualified_reports = []
    
    for item in raw_items:
        # 去重
        if item['title'] in history_ids:
            continue
            
        print(f"正在分析: [{item['source']}] {item['title'][:20]}...")
        result = analyze_with_llm(item)
        
        # 只有高于最低分的才收录
        if result['score'] >= CONFIG['MIN_SCORE']:
            item['score'] = result['score']
            item['summary'] = result['summary']
            item['fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            # 生成一个简短ID供前端使用
            item['id'] = datetime.datetime.now().strftime("%Y%m%d") + "_" + str(len(new_qualified_reports))
            
            new_qualified_reports.append(item)
            
            # 如果凑够了当天的最大数量，就停止（省钱省时间）
            if len(new_qualified_reports) >= CONFIG['FINAL_SAVE_COUNT']:
                break
    
    # 按分数从高到低排序
    new_qualified_reports.sort(key=lambda x: x['score'], reverse=True)

    # 4. 如果有新内容，执行保存和推送
    if new_qualified_reports:
        print(f">>> 发现 {len(new_qualified_reports)} 条优质内容，正在保存和推送...")
        
        # A. 保存到 JSON (供 Geeker Admin 前端使用)
        if os.path.exists(CONFIG["DATA_FILE"]):
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        else:
            current_data = []
            
        # 合并并保留最新的 N 条
        final_data = new_qualified_reports + current_data
        final_data = final_data[:CONFIG['MAX_HISTORY']]
        
        with open(CONFIG["DATA_FILE"], 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        # B. 钉钉/飞书 推送 (只推分数最高的 Top 5)
        top_picks = [r for r in new_qualified_reports if r['score'] >= CONFIG['PUSH_THRESHOLD']]
        if top_picks:
            ding_md = "# 📅 今日量化情报\n\n"
            for r in top_picks[:5]: # 最多推5条
                ding_md += f"### {r['title']}\n"
                ding_md += f"**{r['score']}分** | {r['source']} | {r['broker']}\n"
                ding_md += f"> {r['summary']}\n"
                ding_md += f"[📄 点击阅读 PDF]({r['url']})\n\n---\n"
            send_dingtalk(ding_md)

        # C. 邮件推送 (推所有符合要求的)
        email_html = "<h2>📅 今日量化情报汇总</h2><hr>"
        for r in new_qualified_reports:
            color = "red" if r['score'] >= 8 else "black"
            email_html += f"""
            <div style='margin-bottom:15px; padding:10px; border-left:4px solid #1890ff; background:#f5f5f5'>
                <h3 style='margin:0'><a href='{r['url']}'>{r['title']}</a> <span style='color:{color}'>({r['score']}分)</span></h3>
                <p style='margin:5px 0; font-size:12px; color:#666'>{r['source']} | {r['broker']} | {r['fetch_date']}</p>
                <p style='margin:5px 0'><strong>AI点评:</strong> {r['summary']}</p>
            </div>
            """
        send_email(f"量化日报 ({datetime.date.today()}) - {len(new_qualified_reports)}篇更新", email_html)
        
    else:
        print(">>> 今日无满足条件的高分内容更新。")

if __name__ == "__main__":
    main()
