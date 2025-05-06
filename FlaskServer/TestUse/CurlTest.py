import requests

def upload_video(file_path, upload_url):
    """
    通过 POST 请求上传视频文件到指定 URL
    :param file_path: 本地视频文件路径（如 "./Sample/trailer.mp4"）
    :param upload_url: 服务端上传接口（如 "http://127.0.0.1:5000/upload"）
    """
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.split('/')[-1], f, 'video/mp4')}  # 构造文件字段
            response = requests.post(upload_url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    
    except FileNotFoundError:
        print(f"Error: 文件 {file_path} 不存在！")
    except requests.exceptions.RequestException as e:
        print(f"Error: 请求失败 - {e}")

if __name__ == '__main__':
    # 配置参数
    VIDEO_PATH = "./Sample/trailer.mp4"  # 视频文件路径
    UPLOAD_URL = "http://127.0.0.1:5000/upload"  # 服务端地址
    
    # 执行上传
    upload_video(VIDEO_PATH, UPLOAD_URL)