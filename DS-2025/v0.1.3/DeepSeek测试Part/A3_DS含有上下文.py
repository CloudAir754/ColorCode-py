# 该程序用于进行deepseek进行简单的访问
#       需要重新更改API
#       输出时带有请求信息
#       带有上下文（使用clear清除上下文）
# 1. 有日志，实时显示，不会存储
# 2. 有上下文
## AIGC声明：DeepSeek v3- 网页


import requests
import logging
from typing import Dict, List, Optional, Any

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1/chat/completions"):
        """
        初始化 DeepSeek 客户端。

        :param api_key: DeepSeek API 密钥。
        :param api_url: DeepSeek API 端点 URL，默认为聊天补全 API。
        """
        self.api_key = api_key
        self.api_url = api_url
        self.session = requests.Session()  # 使用会话以复用连接
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        self.messages = []  # 存储对话上下文

    def get_response(self, prompt: str, model: str = "deepseek-chat", max_tokens: int = 150, temperature: float = 0.7) -> Optional[str]:
        """
        调用 DeepSeek API 获取响应，并维护对话上下文。

        :param prompt: 用户输入的提示文本。
        :param model: 使用的模型名称，默认为 "deepseek-chat"。
        :param max_tokens: 生成文本的最大 token 数，默认为 150。
        :param temperature: 控制生成文本的随机性，默认为 0.7。
        :return: 返回生成的文本，如果出错则返回 None。
        """
        # 将用户输入添加到上下文
        self.messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': model,
            'messages': self.messages,  # 包含上下文的对话记录
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        try:
            logger.info(f"Sending request to {self.api_url} with data: {data}")
            response = self.session.post(self.api_url, json=data)
            response.raise_for_status()  # 如果响应状态码不是 200，抛出异常

            result = response.json()
            logger.info(f"Received response: {result}")

            # 提取生成的文本
            if 'choices' in result and len(result['choices']) > 0:
                assistant_reply = result['choices'][0]['message']['content'].strip()
                # 将助手的回复添加到上下文
                self.messages.append({'role': 'assistant', 'content': assistant_reply})
                return assistant_reply
            else:
                logger.error("No choices found in the response.")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except KeyError as e:
            logger.error(f"Invalid response format: {e}")
            return None

    def clear_context(self):
        """清空对话上下文。"""
        self.messages = []

    def close(self):
        """关闭会话，释放资源。"""
        self.session.close()

def main():
    # 替换为你的 DeepSeek API 密钥
    DEFAULT_API_KEY = 'Change with Your apikey'
    # 提示用户输入 API 密钥
    api_key = input("请输入你的 DeepSeek API 密钥（直接按回车使用默认密钥）：").strip()
    if not api_key:
        api_key = DEFAULT_API_KEY
        print("使用默认 API 密钥。")
    client = DeepSeekClient(api_key=api_key)

    print("DeepSeek Chatbot (Type 'exit' or 'quit' to end, 'clear' to reset context)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear':
            client.clear_context()
            print("Context cleared.")
            continue

        response = client.get_response(user_input)
        if response:
            print(f"DeepSeek: {response}")
        else:
            print("Sorry, an error occurred while processing your request.")

    client.close()

if __name__ == "__main__":
    main()