import os
import datetime
import time
import feedparser
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.header import Header
from openai import OpenAI

# ==========================================
#              1. 配置区域
# ==========================================

CONFIG = {
    # --- 顶级媒体 RSS 源 (精选) ---
    "RSS_FEEDS": {
        "Reuters_Business": "https://feeds.reuters.com/reuters/businessNews",
        "Reuters_Tech": "https://feeds.reuters.com/reuters/technologyNews",
        "WSJ_Market": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "FT_World": "https://www.ft.com/?format=rss",
        "Caixin_Biz": "https://www.caixinglobal.com/upload/rss/business_xml.xml",
        "Yahoo_Finance": "https://finance.yahoo.com/news/rssindex", # 很好的聚合源
        # 很多投行研报不公开，但 Seeking Alpha 会有类似的分析
        "Seeking_Alpha": "https://seekingalpha.com/market_currents.xml" 
    },
    
    "LLM_MODEL": "deepseek-chat",
    "MAX_NEWS_COUNT": 40,  # 每天最多处理 40 条新闻给 LLM 总结
}

# ==========================================
#              2. 客户端设置
# ==========================================
LLM_API_KEY = os.environ.get("LLM_API_KEY")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")

client = OpenAI(api_key=LLM_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
#              3. 核心功能函数
# ==========================================

def fetch_rss_news():
    """抓取所有 RSS 源，并按时间过滤出过去 24 小时的内容"""
    print(">>> 开始抓取全球财经 RSS...")
    all_news = []
    
    # 获取 24 小时前的时间戳
    one_day_ago = time.time() - 24 * 3600
    
    for source_name, url in CONFIG["RSS_FEEDS"].items():
        try:
            print(f"正在抓取: {source_name} ...")
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                # 尝试获取发布时间
                published_time = 0
                if hasattr(entry, 'published_parsed'):
                    published_time = time.mktime(entry.published_parsed)
                elif hasattr(entry, 'updated_parsed'):
                    published_time = time.mktime(entry.updated_parsed)
                
                # 过滤旧新闻
                if published_time < one_day_ago:
                    continue
                
                all_news.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": getattr(entry, 'summary', ''),
                    "source": source_name,
                    "time": datetime.datetime.fromtimestamp(published_time).strftime('%Y-%m-%d %H:%M')
                })
        except Exception as e:
            print(f"源 {source_name} 抓取失败: {e}")
            
    print(f">>> 共抓取到 {len(all_news)} 条 24h 内的新闻")
    return all_news

def generate_market_briefing(news_list):
    """让 LLM 阅读所有新闻标题，写一份简报"""
    if not news_list:
        return None
    
    # 截取前 N 条，防止 Token 溢出
    target_news = news_list[:CONFIG["MAX_NEWS_COUNT"]]
    
    # 构造给 LLM 看的文本块
    news_text = ""
    for idx, n in enumerate(target_news):
        news_text += f"{idx+1}. [{n['source']}] {n['title']}\n"
    
    prompt = f"""
    你是一名华尔街资深宏观分析师。请阅读以下过去24小时的全球财经新闻标题：
    
    {news_text}
    
    任务：请撰写一份《每日全球市场情报》，包含以下部分（使用中文）：
    1. **市场情绪评分** (0-10分, 0恐慌/10贪婪)：并简述理由。
    2. **核心宏观事件**：总结最重要的3个宏观驱动因素（如美联储动态、地缘政治、中国政策）。
    3. **关键行业动态**：科技/AI、能源、金融等板块的异动。
    4. **风险提示**：需要交易员立刻警惕的黑天鹅信号。
    
    要求：语言简练专业，像彭博终端的早报一样。不要罗列新闻，要“综合分析”。
    """
    
    try:
        response = client.chat.completions.create(
            model=CONFIG["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 # 低一点，保持客观
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 分析失败: {e}")
        return "生成简报失败，请检查日志。"

def send_dingtalk(msg_markdown):
    if not DINGTALK_WEBHOOK: return
    try:
        requests.post(DINGTALK_WEBHOOK, json={
            "msgtype": "markdown", 
            "markdown": {"title": "财经早报", "text": msg_markdown}
        })
    except: pass

def send_email(subject, text_content):
    if not EMAIL_USER or not EMAIL_PASS: return
    try:
        # 这里为了简单，直接发 LLM 生成的纯文本/Markdown 即可
        # 如果需要 HTML 格式，可以把 text_content 转换一下，这里直接用 plain text 也可以
        msg = MIMEText(text_content, 'plain', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_USER
        smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)
        smtp.quit()
    except: pass

# ==========================================
#              4. 主入口
# ==========================================

def main():
    print(">>> 财经新闻任务开始")
    
    # 1. 抓取
    news_items = fetch_rss_news()
    
    if not news_items:
        print("今日无新闻更新")
        return

    # 2. 分析 (生成简报)
    print(">>> 正在生成 AI 简报...")
    briefing = generate_market_briefing(news_items)
    
    # 3. 发送
    # 构造钉钉消息
    ding_text = "# 🌍 全球宏观早报\n\n" + briefing + "\n\n---\n> 数据来源：Reuters, WSJ, FT, Caixin..."
    send_dingtalk(ding_text)
    
    # 构造邮件消息
    email_text = briefing + "\n\n========================\n新闻源列表:\n"
    for n in news_items[:20]:
        email_text += f"- {n['time']} | {n['title']} ({n['source']})\n"
        
    send_email(f"全球财经早报 ({datetime.date.today()})", email_text)
    
    print(">>> 任务完成")

if __name__ == "__main__":
    main()
