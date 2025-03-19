# 该程序用于进行deepseek进行简单的访问
#       需要重新更改API
#       输出不带有请求信息
#       带有上下文（使用clear清除上下文）
# 1. 有日志，不实时显示，存储
# 2. 有上下文
## AIGC声明：DeepSeek v3- 网页


import requests
import logging
import time
from typing import Optional

# 默认的 API 密钥和端点
DEFAULT_API_KEY = 'Change with Your apikey'
API_URL = 'https://api.deepseek.com/v1/chat/completions'  # 确认正确的 API 端点

class DeepSeekClient:
    def __init__(self, api_key: str, api_url: str = API_URL):
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
            logging.info(f"Sending request to {self.api_url} with data: {data}")
            response = self.session.post(self.api_url, json=data)
            response.raise_for_status()  # 如果响应状态码不是 200，抛出异常

            result = response.json()
            logging.info(f"Received response: {result}")

            # 提取生成的文本
            if 'choices' in result and len(result['choices']) > 0:
                assistant_reply = result['choices'][0]['message']['content'].strip()
                # 将助手的回复添加到上下文
                self.messages.append({'role': 'assistant', 'content': assistant_reply})
                return assistant_reply
            else:
                logging.error("No choices found in the response.")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
        except KeyError as e:
            logging.error(f"Invalid response format: {e}")
            return None

    def clear_context(self):
        """清空对话上下文。"""
        self.messages = []
        logging.info("Context cleared.")

    def close(self):
        """关闭会话，释放资源。"""
        self.session.close()
        logging.info("Session closed.")

def setup_logging():
    """
    配置日志记录，将日志保存到以时间戳命名的文件中。
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"deepseek_chatbot_{timestamp}.log"
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # 创建日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # 添加文件处理器到日志记录器
    logger.addHandler(file_handler)

def main():
    # 配置日志记录
    setup_logging()

    # 提示用户输入 API 密钥
    api_key = input("请输入你的 DeepSeek API 密钥（直接按回车使用默认密钥）：").strip()
    if not api_key:
        api_key = DEFAULT_API_KEY
        print("使用默认 API 密钥。")
        logging.info("Using default API key.")

    client = DeepSeekClient(api_key=api_key)

    print("DeepSeek Chatbot (输入 'exit' 或 'quit' 退出, 输入 'clear' 清空上下文)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            logging.info("Exiting program.")
            break
        elif user_input.lower() == 'clear':
            client.clear_context()
            print("上下文已清空。")
            continue

        response = client.get_response(user_input)
        if response:
            print(f"DeepSeek: {response}")
        else:
            print("抱歉，处理你的请求时出错了。")

    client.close()

if __name__ == "__main__":
    main()