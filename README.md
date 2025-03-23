# 接受POST的视频

- 该服务器是一个工具，接受通过POST发送的视频，将视频拆散为单独的帧。
- 该工具将会在远期与[ColorCode-py](https://github.com/CloudAir754/ColorCode-py)进行配合

# 1. 启用方法
分为启动服务器、发送视频两个部分。
## 1.1 启动服务器
```
python run.py
```
启动程序
## 1.2 发送测试视频
```
curl -X POST -F "file=@./Sample/trailer.mp4" http://localhost:5000/upload
```
用`curl`命令，以post方式，发送视频到指定路径


# 2. Requirements
```
pip freeze > requirements.txt
# 获取所有软件包列表
pip install pipreqs
pipreqs .
# 获取当前使用软件包列表
```