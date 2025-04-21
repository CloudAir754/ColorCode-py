from flask import request, jsonify
import tempfile
import os
from .video_processor import process_video
from werkzeug.utils import secure_filename
import time
import uuid
import glob
from collections import defaultdict

# 存储分片和任务信息的临时目录
# 统一用 os.path.join 创建路径（兼容所有操作系统）
UPLOAD_TEMP_DIR = os.path.join("out", "202504")
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# 存储每个文件ID已接收的分片
received_chunks = defaultdict(set)

from collections import defaultdict
import threading

# 存储每个文件ID的上传状态
upload_status = defaultdict(dict)
# 线程锁，保证线程安全
upload_lock = threading.Lock()

def init_routes(app):
    @app.route('/upload', methods=['POST'])
    def upload_video():
        """处理视频分片上传"""
        # 获取分片信息
        chunk_number = request.form.get('chunk_number', type=int)
        total_chunks = request.form.get('total_chunks', type=int)
        file_id = request.form.get('file_id')
        original_filename = secure_filename(request.form.get('original_filename', ''))
        
        if 'file' not in request.files:
            print("[Error]No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            print("[Error]Empty filename in request")
            return jsonify({"error": "No selected file"}), 400

        # 使用线程锁保证线程安全
        with upload_lock:
            # 如果是第一个分片（无论哪个编号），初始化记录
            if file_id not in upload_status:
                upload_status[file_id] = {
                    'total_chunks': total_chunks,
                    'received_chunks': set(),
                    'created_at': time.time()
                }
                print(f"Starting new upload session, file_id: {file_id}")
            elif upload_status[file_id]['total_chunks'] != total_chunks:
                print(f"[Error]Total chunks mismatch for file {file_id}")
                return jsonify({"error": "Total chunks mismatch"}), 400
            
            # 保存分片到临时目录
            chunk_dir = os.path.join(UPLOAD_TEMP_DIR, file_id)
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_number}")
            file.save(chunk_path)
            print(f"Saved chunk {chunk_number+1}/{total_chunks} to {chunk_path}")
            
            # 记录已接收的分片
            upload_status[file_id]['received_chunks'].add(chunk_number)
            received_count = len(upload_status[file_id]['received_chunks'])
            
            # 检查是否所有分片都已接收
            if received_count == total_chunks:
                print(f"All chunks received for file {file_id}, merging...")
                
                # 合并所有分片（按编号顺序）
                final_path = os.path.join(UPLOAD_TEMP_DIR, f"{file_id}.mp4")
                with open(final_path, 'wb') as outfile:
                    for i in range(total_chunks):
                        chunk_path = os.path.join(chunk_dir, f"chunk_{i}")
                        with open(chunk_path, 'rb') as infile:
                            outfile.write(infile.read())
                
                # 清理分片文件和状态记录
                for f in glob.glob(os.path.join(chunk_dir, "chunk_*")):
                    os.remove(f)
                os.rmdir(chunk_dir)
                del upload_status[file_id]
                
                print(f"File merged successfully at {final_path}")
                
                # 生成任务ID
                task_id = str(uuid.uuid4())
                print(f"Video processing task created, task_id: {task_id}")
                return jsonify({
                    "message": "Video uploaded and merged successfully",
                    "task_id": task_id,
                    "file_id": file_id
                }), 200
        
        # 对于未完成的分片上传，返回成功响应
        return jsonify({
            "message": "Chunk uploaded successfully",
            "chunk_number": chunk_number,
            "file_id": file_id,
            "received_chunks": received_count,
            "total_chunks": total_chunks
        }), 200
        
    
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
    
    