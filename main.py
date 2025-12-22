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
# ==========================================

CONFIG = {
    # --- 基础设置 ---
    "DATA_FILE": "data/reports.json", 
    "MAX_HISTORY": 500,               
    "MIN_SCORE": 5.0,                 # 门槛分
    "PUSH_THRESHOLD": 6.0,            # 推送分
    "FINAL_SAVE_COUNT": 15,           
    
    # --- ArXiv (国际论文) 设置 ---
    "FETCH_COUNT_ARXIV": 30,          
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
    
    # --- Google Scholar 设置 ---
    "GOOGLE_QUERIES": [
        'quantitative trading "reinforcement learning" after:2024', 
        'quantitative trading "deep learning" after:2024',          
        '"algorithmic trading" strategy after:2024'                 
    ],
    "FETCH_COUNT_GOOGLE_PER_QUERY": 5, 
}

# ==========================================
#              2. 环境变量加载
# ==========================================
LLM_API_KEY = os.environ.get("LLM_API_KEY")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

# 初始化 AI (DeepSeek)
client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
#              3. 核心抓取逻辑
# ==========================================

def fetch_arxiv():
    """抓取 ArXiv (修复版：适配 arxiv 库新版本)"""
    print(f"--- 正在抓取 ArXiv ---")
    
    # 构造查询语句
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
            # === 修复点开始 ===
            # 新版 arxiv 库中，r.categories 本身就是 ['q-fin.CP', 'cs.AI'] 这样的字符串列表
            # 所以直接判断字符串即可，不需要 .term
            if not any(tag.startswith(('q-fin', 'cs', 'stat')) for tag in r.categories):
                continue
            # === 修复点结束 ===

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
        # 为了调试，打印一下错误详情
        import traceback
        traceback.print_exc()
        return []

def fetch_google_scholar():
    """抓取 Google Scholar"""
    if not SERPAPI_KEY:
        print("未配置 SERPAPI_KEY，跳过")
        return []
        
    print(f"--- 正在抓取 Google Scholar ---")
    all_results = []
    
    for query in CONFIG['GOOGLE_QUERIES']:
        try:
            print(f"搜索: {query} ...")
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": CONFIG['FETCH_COUNT_GOOGLE_PER_QUERY'],
                "hl": "en" # 强制英文结果
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
#              4. 智能分析与翻译 (核心修改)
# ==========================================

def analyze_with_llm(item):
    """
    调用 AI：
    1. 评分
    2. 翻译摘要为中文
    """
    try:
        # --- 这里的 Prompt 修改了 ---
        prompt = f"""
        你是一名资深的量化交易研究员。请阅读以下英文论文的标题和摘要。
        
        任务：
        1. 评分 (0-10分)："强化学习/深度学习+交易"类论文给8分以上，纯理论数学给4分以下。
        2. 中文摘要：请将英文摘要翻译成通俗流畅的中文。翻译时请保留关键的算法名称（如 Transformer, LSTM, PPO 等）不翻译，确保量化同行能看懂。
        
        论文标题: {item['title']}
        原文摘要: {item['abstract'][:1200]}
        
        请严格按 JSON 格式返回：
        {{
            "score": <数字>,
            "summary": "<这里填中文翻译内容，不要只写一句话，要完整概括核心逻辑，字数控制在100-300字>"
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
    """发送钉钉消息 (调试版)"""
    if not DINGTALK_WEBHOOK: 
        print(">>> 警告: 未配置 DINGTALK_WEBHOOK，跳过钉钉推送")
        return
    
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": "量化日报推送", # 注意：如果你的关键词设为'量化'，这个标题能命中
                "text": msg_markdown
            }
        }
        
        # 发送请求
        response = requests.post(DINGTALK_WEBHOOK, json=data)
        
        # === 关键修改：打印钉钉服务器的回复 ===
        print(f"钉钉发送状态码: {response.status_code}")
        print(f"钉钉响应内容: {response.text}")
        
    except Exception as e:
        print(f"钉钉请求发生异常: {e}")

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
#              5. 主程序入口 (逻辑修复版)
# ==========================================

def main():
    print(">>> 任务开始")
    
    # 1. 加载历史 (避免重复)
    history_ids = []
    if os.path.exists(CONFIG["DATA_FILE"]):
        try:
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                history_ids = [item.get('title') for item in old_data]
        except: pass

    # 2. 抓取 (ArXiv + Scholar)
    # 建议：为了防止 ArXiv 太多，可以把 Scholar 放在前面，或者打乱顺序
    # 但最稳妥的还是“全部分析，最后排序”
    raw_items = []
    raw_items += fetch_google_scholar() # 把 Scholar 放到前面抓
    raw_items += fetch_arxiv()
    
    print(f">>> 共抓取到 {len(raw_items)} 条，开始 AI 翻译与评分...")
    print(">>> 注意：将对所有内容进行打分，这可能需要 1-2 分钟...")

    # 3. AI 分析 (移除 break 限制，让大家公平竞争)
    processed_items = []
    
    for item in raw_items:
        # 去重
        if item['title'] in history_ids: 
            continue
            
        print(f"正在分析: [{item['source']}] {item['title'][:30]}...")
        result = analyze_with_llm(item)
        
        # 只要分数达标，先存进临时列表
        if result['score'] >= CONFIG['MIN_SCORE']:
            item['score'] = result['score']
            item['summary'] = result['summary']
            item['fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            # 临时 ID
            item['id'] = datetime.datetime.now().strftime("%Y%m%d") + "_" + str(len(processed_items))
            
            processed_items.append(item)
            
    # 4. 排序与截断 (关键修改)
    # 先按分数从高到低排序
    processed_items.sort(key=lambda x: x['score'], reverse=True)
    
    # 然后只取前 N 名 (精英策略)
    new_qualified = processed_items[:CONFIG['FINAL_SAVE_COUNT']]
    
    print(f">>> 经 AI 筛选，共有 {len(new_qualified)} 条入选今日日报")

    # 5. 推送逻辑
    if new_qualified:
        # A. 保存到 JSON
        if os.path.exists(CONFIG["DATA_FILE"]):
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                current = json.load(f)
        else: current = []
        
        # 把最新的插到最前面
        final_data = new_qualified + current
        with open(CONFIG["DATA_FILE"], 'w', encoding='utf-8') as f:
            json.dump(final_data[:CONFIG['MAX_HISTORY']], f, ensure_ascii=False, indent=2)

        # B. 钉钉推送 (推送前 5 条)
        top_picks = [r for r in new_qualified if r['score'] >= CONFIG['PUSH_THRESHOLD']]
        # 如果虽然入选了，但分数都没达到推送门槛，就不推钉钉
        if top_picks:
            # 取前 5 个，或者全部 top_picks 中较少的那个
            push_list = top_picks[:5]
            
            ding_md = "# 📅 今日量化论文摘要\n\n"
            for r in push_list:
                ding_md += f"### {r['title']}\n"
                ding_md += f"**{r['score']}分** | {r['source']}\n\n"
                ding_md += f"> **中文摘要**：\n> {r['summary']}\n\n"
                ding_md += f"[📄 原文链接]({r['url']})\n\n---\n"
            send_dingtalk(ding_md)
        else:
            print(">>> 虽然有入选文章，但分数均未达到推送门槛 (PUSH_THRESHOLD)，跳过钉钉。")

        # C. 邮件推送 (推送所有入选的，最多 15 条)
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
