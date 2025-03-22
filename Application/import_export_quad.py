import json
import numpy as np
import os

def import_quadrilaterals(self):
    """从 JSON 文件导入四边形数据"""
    if not self.quad_file_path:
        print("未提供四边形文件路径，跳过导入。")
        return

    try:
        # 从 JSON 文件读取数据
        with open(self.quad_file_path, 'r') as f:
            quadrilaterals_list = json.load(f)

        # 将列表数据转换回 NumPy 数组
        self.quadrilaterals = [np.array(quad) for quad in quadrilaterals_list]
        print(f"四边形数据已成功从 {self.quad_file_path} 导入")
    except FileNotFoundError:
        print(f"文件 {self.quad_file_path} 不存在，跳过导入。")
    except Exception as e:
        print(f"导入四边形数据时出错: {e}")


def export_quadrilaterals(self):
    """导出检测到的四边形数据到 JSON 文件"""
    if self.use_provided_quad or not self.quad_file_path:
        print("未提供四边形文件路径或已使用提供的四边形数据，跳过导出。")
        return

    # 确保目录存在
    os.makedirs(os.path.dirname(self.quad_file_path), exist_ok=True)

    try:
        # 将 NumPy 数组转换为列表
        quadrilaterals_list = [quad.tolist() for quad in self.quadrilaterals]

        # 导出为 JSON 文件
        with open(self.quad_file_path, 'w') as f:
            json.dump(quadrilaterals_list, f, indent=4)
        print(f"四边形数据已成功导出到 {self.quad_file_path}")
    except Exception as e:
        print(f"导出四边形数据时出错: {e}")

