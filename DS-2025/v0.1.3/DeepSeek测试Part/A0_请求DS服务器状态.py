# 该程序用于向status页请求服务器状态
#   可作为调用API，却返回超时后，自动筛查的脚本
## AIGC声明：DeepSeek v3- 网页

import requests

def check_deepseek_status():
    url = "https://status.deepseek.com/api/v2/status.json"
    # 2025/03 可用
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        
        data = response.json()
        
        # 解析并打印状态信息
        status = data.get('status', {})
        description = status.get('description', 'No description available')
        indicator = status.get('indicator', 'No indicator available')
        # 良好的时候，该json返回的全文是；该报文是2025/03/12 15：15 东八区时区申请的。
        '''
        {
            "page": {
                "id": "0db0rq26tg1l",
                "name": "DeepSeek Service",
                "url": "https://status.deepseek.com",
                "time_zone": "Asia/Shanghai",
                "updated_at": "2025-03-12T10:59:30.245+08:00"
            },
            "status": {
                "indicator": "none",
                "description": "All Systems Operational"
            }
        }                
        '''
        
        print(f"Current DeepSeek Status: {indicator}")
        print(f"Description: {description}")
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_deepseek_status()