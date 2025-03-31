from flask import Flask

# 创建 Flask 应用实例
app = Flask(__name__)

# 导入并初始化路由
from .routes import init_routes
init_routes(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)