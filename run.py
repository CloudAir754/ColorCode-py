from app import app

if __name__ == '__main__':
    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=True)