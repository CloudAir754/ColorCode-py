# 该程序用于进行deepseek进行简单的访问
#       需要重新更改API
# 1. 无日志
# 2. 无上下文（即每一次问答互不相关）
## AIGC声明：DeepSeek v3- 网页


import requests

# 替换为你的 DeepSeek API 密钥
DEFAULT_API_KEY = 'Change with Your apikey'
API_URL = 'https://api.deepseek.com/v1/chat/completions'  # 确认正确的 API 端点

def get_deepseek_response(user_input,API_KEY):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'deepseek-chat',  # 模型名称
        'messages': [
            {'role': 'user', 'content': user_input}
        ],
        'max_tokens': 150,  # 返回token最大值
        'temperature': 0.7 # 温度，代码类使用0，数据分析1，创意、诗歌1.5
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

def main():
    # 提示用户输入 API 密钥
    api_key = input("请输入你的 DeepSeek API 密钥（直接按回车使用默认密钥）：").strip()
    if not api_key:
        api_key = DEFAULT_API_KEY
        print("使用默认 API 密钥。")

    print("DeepSeek Chatbot (输入 'exit' 或 'quit' 退出)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        response = get_deepseek_response(user_input,api_key)
        print(f"DeepSeek: {response}")

if __name__ == "__main__":
    main()