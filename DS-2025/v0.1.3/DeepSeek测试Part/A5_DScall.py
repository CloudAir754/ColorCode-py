# 输入颜色，返回列表
import requests
import json

# 替换为你的 DeepSeek API 密钥
DEFAULT_API_KEY = 'sk-c0e3a90a15ca49c383f7dc722c31771a'
API_URL = 'https://api.deepseek.com/v1/chat/completions'  # 确认正确的 API 端点

def get_deepseek_response(user_input,API_KEY):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'deepseek-chat',  # 模型名称
        'messages': [
            #{'role': 'system', 'content': '接收opencv生成的hsv颜色列表，从“红色”、“绿色”、“蓝色”、“黑色”中选择回答；返回格式为json，标签为colors；每组用hsv复述hsv，color存颜色中文。注意opencv下h通道最大值180'},
            {'role': 'system', 'content': '接收opencv生成的RGB颜色列表，从“红色”、“绿色”、“蓝色”、“黑色”中选择回答；返回格式为json，标签为colors；每组用RGB复述RGB，color存颜色中文。'},
            {'role': 'user', 'content': user_input}
        ],
        'max_tokens': 150,  # 返回token最大值
        'temperature': 1 ,# 温度，代码类使用0，数据分析1，创意、诗歌1.5
        'response_format':{
              'type': 'json_object' # 以json格式返回数据  
              # https://api-docs.deepseek.com/zh-cn/guides/json_mode
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"


if __name__ == "__main__":
    response = get_deepseek_response("[#ddbdff]",DEFAULT_API_KEY)
    # 测试结果表明，最好改成最大值都是1的hsv
    # HSV色盘 https://lab.pyzy.net/palette.html
    # 试试rgb？
    
    data = json.loads(response)
    colorlist = data['colors']
    print(colorlist)