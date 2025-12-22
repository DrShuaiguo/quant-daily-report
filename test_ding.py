import os
import requests

# 直接从环境变量获取 Webhook
WEBHOOK_URL = os.environ.get("DINGTALK_WEBHOOK")

def test_send():
    if not WEBHOOK_URL:
        print("❌ 错误：环境变量中没有找到 DINGTALK_WEBHOOK")
        return

    print(f"正在尝试发送到: {WEBHOOK_URL[:20]}...")

    # 构造最简单的消息
    # 注意：这里的标题和内容都包含了“量化”和“日报”，为了命中你的安全关键词
    data = {
        "msgtype": "markdown",
        "markdown": {
            "title": "量化日报测试",
            "text": "# 🔔 这是一个测试\n\n如果你的钉钉机器人设置了关键词“量化”或“日报”，你应该能看到这条消息。\n\n**发送时间**: 现在"
        }
    }

    try:
        resp = requests.post(WEBHOOK_URL, json=data)
        print(f"📡 状态码: {resp.status_code}")
        print(f"📩 响应内容: {resp.text}")
        
        if resp.json().get("errcode") == 0:
            print("✅ 发送成功！请看手机！")
        else:
            print("❌ 发送失败！请根据上面的响应内容排查错误码。")
            
    except Exception as e:
        print(f"💥 代码报错: {e}")

if __name__ == "__main__":
    test_send()
