import subprocess
import os

def delete_2025_files_and_folders():
    target_dir = "./out"  # 目标目录（可修改）
    
    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误：目录 {target_dir} 不存在！")
        return
    
    # 构造 rm -rf 命令（兼容 Linux/macOS 和 Windows Git Bash）
    if os.name == "posix":  # Linux/macOS
        command = f"rm -rf {target_dir}/2025*"
    else:  # Windows（假设使用 Git Bash）
        command = f"rm -rf {target_dir}/2025*"
    
    # 执行命令
    try:
        print(f"执行命令: {command}")
        subprocess.run(command, shell=True, check=True)
        print("删除成功！")
    except subprocess.CalledProcessError as e:
        print(f"删除失败: {e}")

if __name__ == "__main__":
    # 确认操作（防止误删）

    confirm = input("即将删除 ./out/2025* 下的所有文件和文件夹，是否继续？(y/n): ")
    if confirm.lower() == "y":
        delete_2025_files_and_folders()
    else:
        print("操作已取消")