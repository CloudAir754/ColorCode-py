import os
import shutil

def convert_py_to_txt(folder_path):
    # 创建txt文件夹路径
    txt_folder = os.path.join(folder_path, 'txt')
    
    # 如果txt文件夹不存在，则创建它
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    
    # 获取文件夹中的所有.py文件
    py_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]
    
    # 转换文件并保存到txt文件夹
    converted_files = []
    for py_file in py_files:
        # 构建源文件和目标文件的完整路径
        source_file = os.path.join(folder_path, py_file)
        target_file = os.path.join(txt_folder, py_file.replace('.py', '.txt'))
        
        # 复制并重命名文件
        shutil.copy2(source_file, target_file)
        converted_files.append(target_file)
    
    # 输出转换文件的列表
    print("转换的文件列表：")
    for file in converted_files:
        print(file)

if __name__ == "__main__":
    # 输入文件夹地址
    folder_path = input("请输入文件夹地址: ")
    
    # 调用转换函数
    convert_py_to_txt(folder_path)