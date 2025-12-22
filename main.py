import os
import json
import datetime
import requests
import smtplib
import arxiv
from email.mime.text import MIMEText
from email.header import Header
from openai import OpenAI
from serpapi import GoogleSearch

# ==========================================
#              1. 全局配置区域 (CONFIG)
#           ⚙️ 所有参数都在这里修改！
# ==========================================

CONFIG = {
    # --- 基础路径与阈值 ---
    "DATA_FILE": "data/reports.json", 
    "MAX_HISTORY": 500,               # 历史记录保留条数
    "MIN_SCORE": 5.0,                 # 收录门槛 (低于此分直接丢弃)
    "PUSH_THRESHOLD": 6.0,            # 推送门槛 (高于此分才发钉钉)
    
    # --- 数量控制 (No Magic Numbers!) ---
    "FINAL_SAVE_COUNT": 15,           # 每天最终收录并发送邮件的最大篇数
    "DINGTALK_PUSH_LIMIT": 5,         # 钉钉最多推送几篇 (防止刷屏)
    
    # --- 抓取源设置 ---
    "FETCH_COUNT_ARXIV": 30,          # ArXiv 原始抓取量
    "FETCH_COUNT_GOOGLE_PER_QUERY": 5,# Google 每个关键词抓取量
    
    # --- 文本处理 ---
    "MAX_TEXT_LENGTH_FOR_AI": 1200,   # 喂给 AI 的摘要最大长度 (字符数)
    
    # --- 搜索时间范围 ---
    "SEARCH_YEAR": "2024",            # 搜索哪一年之后的文章
    
    # --- ArXiv 关键词 ---
    "ARXIV_KEYWORDS": [
        "quantitative finance",
        "factor model",
        "portfolio optimization",
        "deep learning trading",      
        "reinforcement learning trading", 
        "machine learning trading",   
        "algorithm trading",          
        "market microstructure",
        "risk premia"
    ],
    
    # --- Google Scholar 关键词 ---
    # 注意：这里的 year 会在代码里动态替换
    "GOOGLE_QUERIES": [
        'quantitative trading "reinforcement learning"', 
        'quantitative trading "deep learning"',          
        '"algorithmic trading" strategy'                 
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

# 初始化 AI
client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
#              3. 核心抓取逻辑
# ==========================================

def fetch_arxiv():
    """抓取 ArXiv"""
    print(f"--- 正在抓取 ArXiv ---")
    keywords_query = " OR ".join([f'"{k}"' for k in CONFIG['ARXIV_KEYWORDS']])
    query = f'(cat:q-fin.* OR cat:cs.AI) AND ({keywords_query})'
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=CONFIG['FETCH_COUNT_ARXIV'],
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = []
        for r in search.results():
            # 适配新版 arxiv 库: categories 是 list of strings
            if not any(tag.startswith(('q-fin', 'cs', 'stat')) for tag in r.categories):
                continue

            results.append({
                "title": r.title,
                "url": r.pdf_url,
                "source": "ArXiv",
                "date": r.published.strftime("%Y-%m-%d"),
                "abstract": r.summary,
                "broker": "Cornell Univ" 
            })
        print(f"ArXiv 抓取到 {len(results)} 条")
        return results
    except Exception as e:
        print(f"ArXiv 抓取失败: {e}")
        return []

def fetch_google_scholar():
    """抓取 Google Scholar"""
    if not SERPAPI_KEY:
        print("未配置 SERPAPI_KEY，跳过")
        return []
        
    print(f"--- 正在抓取 Google Scholar ---")
    all_results = []
    
    for base_query in CONFIG['GOOGLE_QUERIES']:
        # 动态拼接年份: query + " after:2024"
        query_with_year = f'{base_query} after:{CONFIG["SEARCH_YEAR"]}'
        
        try:
            print(f"搜索: {query_with_year} ...")
            params = {
                "engine": "google_scholar",
                "q": query_with_year,
                "api_key": SERPAPI_KEY,
                "num": CONFIG['FETCH_COUNT_GOOGLE_PER_QUERY'],
                "hl": "en"
            }
            search = GoogleSearch(params)
            data = search.get_dict()
            organic_results = data.get("organic_results", [])
            
            for item in organic_results:
                if 'link' not in item: continue
                all_results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "source": "Scholar",
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "abstract": item.get("snippet", item.get("title")),
                    "broker": "Google Scholar"
                })
        except Exception as e:
            print(f"Scholar 查询出错: {e}")
            
    return all_results

# ==========================================
#              4. 智能分析与翻译
# ==========================================

def analyze_with_llm(item):
    """AI 评分与翻译"""
    try:
        # 使用配置里的长度限制
        abstract_text = item['abstract'][:CONFIG['MAX_TEXT_LENGTH_FOR_AI']]
        
        prompt = f"""
        你是一名资深的量化交易研究员。请阅读以下英文论文的标题和摘要。
        
        任务：
        1. 评分 (0-10分)："强化学习/深度学习+交易"类论文给8分以上，纯理论数学给4分以下。
        2. 中文摘要：请将英文摘要翻译成通俗流畅的中文。翻译时请保留关键的算法名称（如 Transformer, LSTM, PPO 等）不翻译。
        
        论文标题: {item['title']}
        原文摘要: {abstract_text}
        
        请严格按 JSON 格式返回：
        {{
            "score": <数字>,
            "summary": "<这里填中文翻译内容>"
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
        return {"score": 6.0, "summary": "AI 翻译失败，请查看原文。"}

def send_dingtalk(msg_markdown):
    """发送钉钉"""
    if not DINGTALK_WEBHOOK: return
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": "量化日报",
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
    
    # 1. 加载历史
    history_ids = []
    if os.path.exists(CONFIG["DATA_FILE"]):
        try:
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                history_ids = [item.get('title') for item in old_data]
        except: pass

    # 2. 抓取 (Scholar 在前，ArXiv 在后，公平竞争)
    raw_items = []
    raw_items += fetch_google_scholar()
    raw_items += fetch_arxiv()
    
    print(f">>> 共抓取到 {len(raw_items)} 条，开始 AI 翻译与评分...")

    # 3. AI 分析 (无 break 限制，全量跑)
    processed_items = []
    
    for item in raw_items:
        if item['title'] in history_ids: continue
            
        print(f"正在分析: {item['title'][:40]}...")
        result = analyze_with_llm(item)
        
        if result['score'] >= CONFIG['MIN_SCORE']:
            item['score'] = result['score']
            item['summary'] = result['summary']
            item['fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            item['id'] = datetime.datetime.now().strftime("%Y%m%d") + "_" + str(len(processed_items))
            processed_items.append(item)
            
    # 4. 排序与截断 (使用配置参数)
    processed_items.sort(key=lambda x: x['score'], reverse=True)
    
    # 取前 N 名 (使用 CONFIG['FINAL_SAVE_COUNT'])
    new_qualified = processed_items[:CONFIG['FINAL_SAVE_COUNT']]
    
    print(f">>> 经筛选，共有 {len(new_qualified)} 条入选日报")

    # 5. 推送
    if new_qualified:
        # A. 保存
        if os.path.exists(CONFIG["DATA_FILE"]):
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                current = json.load(f)
        else: current = []
        final_data = new_qualified + current
        with open(CONFIG["DATA_FILE"], 'w', encoding='utf-8') as f:
            json.dump(final_data[:CONFIG['MAX_HISTORY']], f, ensure_ascii=False, indent=2)

        # B. 钉钉推送
        top_picks = [r for r in new_qualified if r['score'] >= CONFIG['PUSH_THRESHOLD']]
        if top_picks:
            # 限制钉钉推送数量 (使用 CONFIG['DINGTALK_PUSH_LIMIT'])
            push_limit = CONFIG['DINGTALK_PUSH_LIMIT']
            push_list = top_picks[:push_limit]
            
            ding_md = "# 📅 今日量化论文摘要\n\n"
            for r in push_list:
                ding_md += f"### {r['title']}\n"
                ding_md += f"**{r['score']}分** | {r['source']}\n\n"
                ding_md += f"> **中文摘要**：\n> {r['summary']}\n\n"
                ding_md += f"[📄 原文链接]({r['url']})\n\n---\n"
            send_dingtalk(ding_md)

        # C. 邮件推送
        email_html = "<h2>📅 今日量化交易学术精选</h2><hr>"
        for r in new_qualified:
            color = "red" if r['score'] >= 8 else "black"
            email_html += f"""
            <div style='margin-bottom:20px; padding:15px; border:1px solid #ddd; border-radius:5px;'>
                <h3 style='margin-top:0'><a href='{r['url']}'>{r['title']}</a> <span style='color:{color}'>({r['score']}分)</span></h3>
                <p style='color:#666; font-size:12px'>{r['source']} | {r['date']}</p>
                <div style='background:#f9f9f9; padding:10px; border-left:4px solid #1890ff;'>
                    <p style='margin:0; font-weight:bold;'>🇨🇳 中文摘要：</p>
                    <p style='margin-top:5px; line-height:1.6;'>{r['summary']}</p>
                </div>
            </div>
            """
        send_email(f"量化日报 ({datetime.date.today()}) - {len(new_qualified)}篇 AI 精读", email_html)
        
    else:
        print(">>> 无更新。")

if __name__ == "__main__":
    main()
