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
    "MIN_SCORE": 5.0,                 # 只要 AI 没挂，基本都能过
    "PUSH_THRESHOLD": 6.0,            # 门槛设低点，保证你能收到推送
    "FINAL_SAVE_COUNT": 15,           
    
    # --- ArXiv (国际论文) 设置 ---
    "FETCH_COUNT_ARXIV": 30,          # 抓多点用于过滤
    "ARXIV_KEYWORDS": [
        "quantitative finance",
        "factor model",
        "portfolio optimization",
        "deep learning trading",      # 深度学习
        "reinforcement learning trading", # 强化学习
        "machine learning trading",   # 机器学习
        "algorithm trading",          # 算法交易
        "market microstructure",
        "risk premia"
    ],
    
    # --- Google Scholar (谷歌学术) 设置 ---
    # 这里的逻辑改了：我们会轮询下面这几个查询词，确保覆盖面
    "GOOGLE_QUERIES": [
        'quantitative trading "reinforcement learning" after:2024', # 强化学习+量化
        'quantitative trading "deep learning" after:2024',          # 深度学习+量化
        '"algorithmic trading" strategy after:2024'                 # 算法交易
    ],
    "FETCH_COUNT_GOOGLE_PER_QUERY": 5, # 每个词抓 5 条，总共抓 15 条
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
# 如果要换 Kimi，base_url 改为: "https://api.moonshot.cn/v1"
client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
#              3. 核心抓取逻辑
# ==========================================

def fetch_arxiv():
    """抓取 ArXiv"""
    print(f"--- 正在抓取 ArXiv (关键词: {CONFIG['ARXIV_KEYWORDS'][:3]}...) ---")
    
    # 构造 OR 查询
    keywords_query = " OR ".join([f'"{k}"' for k in CONFIG['ARXIV_KEYWORDS']])
    # 限制分类为 q-fin (量化金融) 或 cs.AI (人工智能)
    query = f'(cat:q-fin.* OR cat:cs.AI) AND ({keywords_query})'
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=CONFIG['FETCH_COUNT_ARXIV'],
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for r in search.results():
            # 简单去重：如果分类完全不沾边，跳过
            # (ArXiv 搜索有时候很宽泛)
            if not any(tag.startswith(('q-fin', 'cs', 'stat')) for tag in [t.term for t in r.categories]):
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
    """抓取 Google Scholar (多关键词轮询版)"""
    if not SERPAPI_KEY:
        print("未配置 SERPAPI_KEY，跳过 Google Scholar")
        return []
        
    print(f"--- 正在抓取 Google Scholar (多轮搜索) ---")
    all_results = []
    
    # 遍历配置里的每一个查询语句
    for query in CONFIG['GOOGLE_QUERIES']:
        try:
            print(f"正在搜 Scholar: {query} ...")
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": CONFIG['FETCH_COUNT_GOOGLE_PER_QUERY'],
                "hl": "en" # 强制英文结果，相关性更高
            }
            search = GoogleSearch(params)
            data = search.get_dict()
            organic_results = data.get("organic_results", [])
            
            if not organic_results:
                print(f"警告: 查询 '{query}' 未返回任何结果")
                continue

            for item in organic_results:
                # 必须要有链接才收录
                if 'link' not in item:
                    continue
                    
                all_results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "source": "Scholar",
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "abstract": item.get("snippet", item.get("title")), # 摘要可能为空
                    "broker": "Google Scholar"
                })
        except Exception as e:
            print(f"Scholar 单次查询出错: {e}")
            continue
            
    print(f"Google Scholar 共抓取到 {len(all_results)} 条")
    return all_results

# ==========================================
#              4. 智能分析与分发
# ==========================================

def analyze_with_llm(item):
    """调用 AI 进行评分和总结"""
    try:
        prompt = f"""
        你是一个专业的量化基金经理。请评估以下学术论文对“实战量化交易”的价值。
        
        标题: {item['title']}
        摘要: {item['abstract'][:800]}
        
        请严格按 JSON 格式返回：
        {{
            "score": <0-10分, 凡是涉及'强化学习/深度学习+交易'的直接给8分以上, 纯理论数学给4分>,
            "summary": "<用中文一句话概括其核心算法或策略模型>"
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
        # 失败时给个默认分，防止因为AI波动漏掉重要论文
        return {"score": 6.0, "summary": "AI 暂时罢工，请人工阅读"}

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
    
    # 1. 加载历史数据
    history_ids = []
    if os.path.exists(CONFIG["DATA_FILE"]):
        try:
            with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                history_ids = [item.get('title') for item in old_data]
        except:
            history_ids = []

    # 2. 抓取 (只抓 ArXiv 和 Google Scholar)
    raw_items = []
    raw_items += fetch_arxiv()
    raw_items += fetch_google_scholar()
    
    print(f">>> 共抓取到 {len(raw_items)} 条原始数据，开始 AI 筛选...")

    # 3. AI 分析与筛选
    new_qualified_reports = []
    
    for item in raw_items:
        # 去重
        if item['title'] in history_ids:
            continue
            
        print(f"正在分析: {item['title'][:40]}...")
        result = analyze_with_llm(item)
        
        if result['score'] >= CONFIG['MIN_SCORE']:
            item['score'] = result['score']
            item['summary'] = result['summary']
            item['fetch_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            item['id'] = datetime.datetime.now().strftime("%Y%m%d") + "_" + str(len(new_qualified_reports))
            
            new_qualified_reports.append(item)
            
            if len(new_qualified_reports) >= CONFIG['FINAL_SAVE_COUNT']:
                break
    
    # 按分数排序
    new_qualified_reports.sort(key=lambda x: x['score'], reverse=True)

    # 4. 保存和推送
    if new_qualified_reports:
        print(f">>> 发现 {len(new_qualified_reports)} 条优质内容，正在推送...")
        
        # A. 保存到 JSON
        if os.path.exists(CONFIG["DATA_FILE"]):
            try:
                with open(CONFIG["DATA_FILE"], 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
            except:
                current_data = []
        else:
            current_data = []
            
        final_data = new_qualified_reports + current_data
        with open(CONFIG["DATA_FILE"], 'w', encoding='utf-8') as f:
            json.dump(final_data[:CONFIG['MAX_HISTORY']], f, ensure_ascii=False, indent=2)

        # B. 钉钉/飞书 推送 (Top 5)
        top_picks = [r for r in new_qualified_reports if r['score'] >= CONFIG['PUSH_THRESHOLD']]
        if top_picks:
            ding_md = "# 📅 今日量化学术日报\n\n"
            for r in top_picks[:5]:
                ding_md += f"### {r['title']}\n"
                ding_md += f"**{r['score']}分** | {r['source']}\n"
                ding_md += f"> {r['summary']}\n"
                ding_md += f"[📄 阅读全文]({r['url']})\n\n---\n"
            send_dingtalk(ding_md)

        # C. 邮件推送
        email_html = "<h2>📅 今日量化交易学术精选</h2><hr>"
        for r in new_qualified_reports:
            color = "red" if r['score'] >= 8 else "black"
            email_html += f"""
            <div style='margin-bottom:15px; padding:10px; border-left:4px solid #52c41a; background:#f6ffed'>
                <h3 style='margin:0'><a href='{r['url']}'>{r['title']}</a> <span style='color:{color}'>({r['score']}分)</span></h3>
                <p style='margin:5px 0; font-size:12px; color:#666'>{r['source']} | {r['date']}</p>
                <p style='margin:5px 0'><strong>AI 解读:</strong> {r['summary']}</p>
            </div>
            """
        send_email(f"量化日报 ({datetime.date.today()}) - {len(new_qualified_reports)}篇更新", email_html)
        
    else:
        print(">>> 今日无满足条件的高分内容更新。")

if __name__ == "__main__":
    main()
