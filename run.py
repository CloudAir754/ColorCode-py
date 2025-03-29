from app import app
from app import delete_testFile

if __name__ == '__main__':
    # 启动 Flask 应用
    delete_testFile.delete_2025_files_and_folders()
    app.run(host='0.0.0.0', port=5000, debug=True)