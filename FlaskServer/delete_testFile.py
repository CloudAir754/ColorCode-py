import subprocess
import os
import shutil

def delete_2025_files_and_folders():
    target_dir = "./out"  # 目标目录（可修改）
    

    # 删除匹配 2025* 的文件和目录
    try:
        for item in os.listdir(target_dir):
            if item.startswith("2025"):
                item_path = os.path.join(target_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)  # 删除文件
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # 删除目录（递归）
        print("删除成功！")
    except Exception as e:
        print(f"删除失败: {e}")

if __name__ == "__main__":
    # 确认操作（防止误删）

    confirm = input("即将删除 ./out/2025* 下的所有文件和文件夹，是否继续？(y/n): ")
    if confirm.lower() == "y":
        delete_2025_files_and_folders()
    else:
        print("操作已取消")