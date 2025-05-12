from FlaskServer import app
from FlaskServer import delete_testFile

if __name__ == '__main__':
    # 启动 Flask 应用
    delete_testFile.delete_2025_files_and_folders()
    from colorama import init, Fore, Back, Style
    init(autoreset=True)  # 自动重置样式

    print(f"{Fore.RED}{'=' * 50}")
    print(f"{Style.BRIGHT}{Fore.YELLOW}电脑请用网线连接，或连接 {Fore.CYAN}BIT-WEB {Fore.YELLOW}无线网络")
    print("注意注意看我看我")
    print(f"{Style.BRIGHT}{Fore.RED}连接 {Fore.MAGENTA}BIT-Mobile {Fore.RED}将无法联网！")
    print(f"{Fore.RED}{'=' * 50}")

    print("===================================================")
    print("电脑请用网线连接，或连接BIT-WEB无线网络")
    print("注意注意看我看我")
    print("连接BIT-Mobie将无法联网！！！！！！")
    print("===================================================")

    # serve(app, host="0.0.0.0", port=5000)  # 替换 app.run() 
    # 不行，不会显示调试信息（找不到IP；对学姐不友好
    app.run(host='0.0.0.0', port=5000, debug=False)