import os
import shutil

def copy_py_to_txt(folder_path):
    # 创建目标目录 'txt'
    txt_folder = os.path.join(folder_path, "txt")
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
        print(f"Created directory: {txt_folder}")

    # 记录已修改的文件列表
    modified_files = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以 .py 结尾
        if filename.endswith(".py"):
            # 构造新的文件名
            new_filename = filename + ".txt"
            # 获取文件的完整路径
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(txt_folder, new_filename)
            # 复制文件到目标目录并重命名
            shutil.copy2(old_file, new_file)
            modified_files.append(new_filename)
            print(f"Copied and renamed {old_file} to {new_file}")

    # 输出已修改的文件列表
    if modified_files:
        print("\nFiles with .txt suffix added:")
        for file in modified_files:
            print(file)
    else:
        print("\nNo .py files found in the directory.")

if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "./ColorCodeDetector/"
    copy_py_to_txt(folder_path)