from flask import request, jsonify
import tempfile
import os
from .video_processor import process_video

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

        # 处理视频
        process_video(temp_file_path)

        # 删除临时文件
        os.remove(temp_file_path)

        return jsonify({"message": "Video processed successfully"}), 200