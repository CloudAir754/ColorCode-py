from flask import request, jsonify
import tempfile
import os
from .video_processor import process_video

import time


def init_routes(app):
    @app.route('/upload', methods=['POST'])
    def upload_video():
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # 保存上传的文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        # 处理视频（先暂时屏蔽）        
        # process_video(temp_file_path)

        # 删除临时文件
        print(temp_file_path)
        # os.remove(temp_file_path)

        return jsonify({"message": "Video processed successfully"}), 200
    
    # 新增测试端口
    @app.route('/ping', methods=['GET'])
    def ping():
        # 返回当前时间戳和服务器状态
        start_time = time.time()
        response = {
            "status": "alive",
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "response_time": None
        }
        response["response_time"] = time.time() - start_time
        return jsonify(response)
    
    