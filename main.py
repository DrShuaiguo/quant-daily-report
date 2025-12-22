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
#         ⚙️ 纯净版：无魔法数字
# ==========================================

CONFIG = {
    # --- 文件路径 ---
    "DATA_FILE": "data/reports.json",      # 【精华库】给前端/邮件看 (只存高分)
    "HISTORY_FILE": "data/history.json",   # 【黑名单】给爬虫去重用 (存所有读过的)
    
    # --- 核心容量控制 ---
    "MAX_HISTORY_SIZE": 3000,         # 黑名单容量 (必须 > 挖掘深度)
    "MAX_REPORT_SIZE": 500,           # 精华库容量 (保留最近500篇高分)
    "MAX_EMAIL_ITEM_LIMIT": 50,       # 邮件保护阀 (防止邮件过大发不出去)
    
    # --- 阈值设置 ---
    "MIN_SCORE": 4.0,                 # 4分以上才有资格进 reports.json
    "PUSH_THRESHOLD": 6.0,            # 6分以上才推钉钉
    
    "FINAL_SAVE_COUNT": 15,           # 每天最多收录 15 篇
    "DINGTALK_PUSH_LIMIT": 5,         # 钉钉只推 Top 5
    
    # --- 抓取设置 ---
    "CANDIDATE_POOL_SIZE": 20,        # 每次必须凑齐 N 篇【未读】文章喂给 AI
    "MAX_SEARCH_DEPTH": 1000,         # ArXiv 最大翻页深度
    
    "FETCH_COUNT_GOOGLE_PER_QUERY": 10, # Google 每个关键词抓 N 条
    
    "MAX_TEXT_LENGTH_FOR_AI": 1200,   # 摘要截断长度
    "SEARCH_YEAR": "2024",            # 搜索年份
    
    # --- 关键词 ---
    "ARXIV_KEYWORDS": [
        "quantitative finance", "factor model", "portfolio optimization",
        "deep learning trading", "reinforcement learning trading", 
        "machine learning trading", "algorithm trading",          
        "market microstructure", "risk premia", "quantitative trading",
        "deep reinforcement learning", "transformer finance",
        "large language model trading"
    ],
    "GOOGLE_QUERIES": [
        'quantitative trading "reinforcement learning"', 
        'quantitative trading "deep learning"',          
        '"algorithmic trading" strategy',
        'transformers for "stock prediction"', 
        '"LLM" agents for "quantitative trading"'
    ]
}

# ==========================================
#              2. 环境与客户端
# ==========================================
LLM_API_KEY = os.environ.get("LLM_API_KEY")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
#              3. 抓取函数
# ==========================================

def fetch_arxiv_smart(history_titles):
    """
    智能抓取 ArXiv: 
    根据 history.json 里的标题进行去重。
    """
    target_count = CONFIG['CANDIDATE_POOL_SIZE']
    print(f"--- ArXiv 智能深挖 (目标: 找到 {target_count} 篇未读) ---")
    
    keywords_query = " OR ".join([f'"{k}"' for k in CONFIG['ARXIV_KEYWORDS']])
    query = f'(cat:q-fin.* OR cat:cs.AI) AND ({keywords_query})'
    
    candidates = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=CONFIG['MAX_SEARCH_DEPTH'], 
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        scanned = 0
        for r in search.results():
            scanned += 1
            if not any(tag.startswith(('q-fin', 'cs', 'stat')) for tag in r.categories): continue
            
            # === 简单去重 ===
            # 只去除首尾空格，不做复杂的大小写转换
            if r.title.strip() in history_titles:
                continue 
                
            candidates.append({
                "title": r.title.strip(), # 存的时候也去一下空格
                "url": r.pdf_url, 
                "source": "ArXiv",
                "date": r.published.strftime("%Y-%m-%d"), 
                "abstract": r.summary,
                "broker": "Cornell Univ" 
            })
            
            if len(candidates) >= target_count:
                print(f"--> 已凑齐 {len(candidates)} 篇未读，停止扫描。")
                break
                
        print(f"扫描结束: 共扫描 {scanned} 篇，筛选出 {len(candidates)} 篇新文章。")
        return candidates
    except Exception as e:
        print(f"ArXiv Error: {e}")
        return []

def fetch_google_scholar():
    if not SERPAPI_KEY: return []
    print(f"--- 正在抓取 Google Scholar ---")
    all_results = []
    for base_query in CONFIG['GOOGLE_QUERIES']:
        try:
            params = {
                "engine": "google_scholar", 
                "q": f'{base_query} after:{CONFIG["SEARCH_YEAR"]}',
                "api_key": SERPAPI_KEY, 
                "num": CONFIG['FETCH_COUNT_GOOGLE_PER_QUERY'], 
                "hl": "en"
            }
            search = GoogleSearch(params)
            for item in search.get_dict().get("organic_results", []):
                if 'link' not in item: continue
                all_results.append({
                    "title": item.get("title").strip(), # 去空格
                    "url": item.get("link"),
                    "source": "Scholar", 
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "abstract": item.get("snippet", item.get("title")), 
                    "broker": "Google Scholar"
                })
        except: pass
    return all_results

# ==========================================
#              4. AI 分析
# ==========================================

def analyze_with_llm(item):
    try:
        prompt = f"""
        你是一名量化交易员。评估以下论文对“实战交易”的价值。
        标题: {item['title']}
        摘要: {item['abstract'][:CONFIG['MAX_TEXT_LENGTH_FOR_AI']]}
        
        1. 评分(0-10): 实战强(RL/DeepLearning/Alpha)给8-10分，纯理论给3-5分，无关给0分。
        2. 中文摘要: 翻译核心，保留术语。
        返回JSON: {{"score": 0, "summary": "..."}}
        """
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except: return {"score": 0, "summary": "Error"}

def send_dingtalk(msg):
    if not DINGTALK_WEBHOOK: return
    try: requests.post(DINGTALK_WEBHOOK, json={"msgtype": "markdown", "markdown": {"title": "量化日报", "text": msg}})
    except: pass

def send_email(subject, html):
    if not EMAIL_USER or not EMAIL_PASS: return
    try:
        msg = MIMEText(html, 'html', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_USER
        smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)
        smtp.quit()
    except: pass

# ==========================================
#              5. 主程序 (Simple & Clean)
# ==========================================

def main():
    print(">>> 任务开始")
    
    # 1. 加载【黑名单】
    history_titles = []
    if os.path.exists(CONFIG["HISTORY_FILE"]):
        try:
            with open(CONFIG["HISTORY_FILE"], 'r', encoding='utf-8') as f:
                history_titles = json.load(f)
        except: pass
        
    print(f"载入历史记录: {len(history_titles)} 条")

    # 本次运行新增的已读标题 (无论分高低，都追加到这里)
    new_analyzed_titles = []
    
    # 本次运行入选的高分文章 (追加到 reports.json)
    qualified_items = []

    # === 阶段一：ArXiv ===
    candidates = fetch_arxiv_smart(history_titles)
    
    for item in candidates:
        if len(qualified_items) >= CONFIG['FINAL_SAVE_COUNT']:
            print(">>> [ArXiv] 今日高分名额已满，停止分析。")
            break
            
        print(f"分析: {item['title'][:30]}...")
        result = analyze_with_llm(item)
        
        # 只要分析过，就记录 (用于去重)
        new_analyzed_titles.append(item['title'])
        
        if result['score'] >= CONFIG['MIN_SCORE']:
            item.update(result)
            item['fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            item['id'] = datetime.datetime.now().strftime("%Y%m%d") + "_" + str(len(qualified_items))
            qualified_items.append(item)

    # === 阶段二：Scholar 补货 (简单版去重) ===
    if len(qualified_items) < CONFIG['FINAL_SAVE_COUNT']:
        needed = CONFIG['FINAL_SAVE_COUNT'] - len(qualified_items)
        print(f">>> Scholar 补货 (缺 {needed} 条)...")
        
        scholar_candidates = fetch_google_scholar()
        for item in scholar_candidates:
            if len(qualified_items) >= CONFIG['FINAL_SAVE_COUNT']: break
            
            # --- 简单去重逻辑 ---
            # 1. 查历史总账
            if item['title'] in history_titles: continue 
            # 2. 查刚才 ArXiv 的账 (防止本次运行撞车)
            if item['title'] in new_analyzed_titles: continue 
            
            print(f"分析: {item['title'][:30]}...")
            result = analyze_with_llm(item)
            
            new_analyzed_titles.append(item['title'])
            
            if result['score'] >= CONFIG['MIN_SCORE']:
                item.update(result)
                item['fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
                item['id'] = datetime.datetime.now().strftime("%Y%m%d") + "_s_" + str(len(qualified_items))
                qualified_items.append(item)

    # === 保存逻辑 ===
    
    # A. 保存 history.json (所有标题，用于去重)
    if new_analyzed_titles:
        final_history = new_analyzed_titles + history_titles
        final_history = final_history[:CONFIG['MAX_HISTORY_SIZE']]
        
        os.makedirs(os.path.dirname(CONFIG["HISTORY_FILE"]), exist_ok=True)
        with open(CONFIG["HISTORY_FILE"], 'w', encoding='utf-8') as f:
            json.dump(final_history, f, ensure_ascii=False, indent=2)
            
    # B. 保存 reports.json (仅高分文章，用于展示)
    if qualified_items:
        qualified_items.sort(key=lambda x: x['score'], reverse=True)
        
        if os.path.exists(CONFIG["DATA_FILE"]):
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                old_reports = json.load(f)
        else: old_reports = []
        
        final_reports = qualified_items + old_reports
        final_reports = final_reports[:CONFIG['MAX_REPORT_SIZE']]
        
        with open(CONFIG["DATA_FILE"], 'w', encoding='utf-8') as f:
            json.dump(final_reports, f, ensure_ascii=False, indent=2)

        # === 推送逻辑 ===
        
        # 1. 钉钉 (Top 5)
        top_picks = [r for r in qualified_items if r['score'] >= CONFIG['PUSH_THRESHOLD']]
        if top_picks:
            push_limit = CONFIG['DINGTALK_PUSH_LIMIT']
            ding_md = "# 📅 量化日报\n\n"
            for r in top_picks[:push_limit]:
                ding_md += f"### {r['title']}\n**{r['score']}分** | {r['source']}\n> {r['summary']}\n[📄 链接]({r['url']})\n\n---\n"
            if len(qualified_items) > push_limit:
                ding_md += f"\n> 💡 还有 {len(qualified_items)-push_limit} 篇已发邮箱。"
            send_dingtalk(ding_md)

        # 2. 邮件 (限制最大条数)
        email_items = qualified_items[:CONFIG['MAX_EMAIL_ITEM_LIMIT']]
        html = f"<h2>量化日报 ({len(qualified_items)}篇)</h2><hr>"
        for r in email_items:
            color = "red" if r['score']>=8 else "black"
            html += f"<div><h3><a href='{r['url']}'>{r['title']}</a> <span style='color:{color}'>({r['score']}分)</span></h3><p>{r['source']} | {r['date']}</p><div style='background:#f9f9f9;padding:10px'>{r['summary']}</div></div><br>"
        
        if len(qualified_items) > len(email_items):
             html += f"<p>... (还有 {len(qualified_items) - len(email_items)} 篇未显示)</p>"

        send_email(f"量化日报 - {len(qualified_items)}篇", html)
        
        print(f">>> 成功更新: 新增历史 {len(new_analyzed_titles)} 条, 新增精华 {len(qualified_items)} 条")
    else:
        print(">>> 无符合条件的新文章。")

if __name__ == "__main__":
    main()
