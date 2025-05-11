import os
import fnmatch
from pathlib import Path

def parse_gitignore(gitignore_path):
    """解析.gitignore文件，返回应忽略的模式列表"""
    ignore_patterns = []
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ignore_patterns.append(line)
    return ignore_patterns

def should_ignore(path, start_path, ignore_patterns, is_dir):
    """检查路径是否应该被忽略"""
    rel_path = os.path.relpath(path, start_path)
    
    # 总是忽略.git目录
    if '.git' in rel_path.split(os.sep):
        return True
    
    # 检查.gitignore模式
    for pattern in ignore_patterns:
        # 处理目录模式（以/结尾）
        if pattern.endswith('/'):
            dir_pattern = pattern.rstrip('/')
            if is_dir and (fnmatch.fnmatch(rel_path, dir_pattern) or 
                          fnmatch.fnmatch(os.path.basename(path), dir_pattern)):
                return True
        else:
            if fnmatch.fnmatch(rel_path, pattern) or \
               fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
    return False

def generate_directory_tree(start_path='.', max_depth=None, indent='    ', 
                          file_output=False, output_file='directory_tree.txt',
                          respect_gitignore=True):
    """
    生成文件夹架构树状图（支持.gitignore）
    
    参数:
        start_path (str): 起始路径，默认为当前文件夹
        max_depth (int): 最大深度，None表示无限制
        indent (str): 缩进字符
        file_output (bool): 是否输出到文件
        output_file (str): 输出文件名
        respect_gitignore (bool): 是否遵守.gitignore规则
    """
    tree = []
    ignore_patterns = []
    start_path = os.path.abspath(start_path)
    
    if respect_gitignore:
        gitignore_path = os.path.join(start_path, '.gitignore')
        ignore_patterns = parse_gitignore(gitignore_path)
    
    def build_tree(path, depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        name = os.path.basename(path)
        
        # 根目录不检查忽略规则
        if depth > 0 and should_ignore(path, start_path, ignore_patterns, os.path.isdir(path)):
            return
        
        prefix = indent * (depth - 1) + '├── ' if depth > 0 else ''
        tree.append(f"{prefix}{name}")
        
        if os.path.isdir(path):
            try:
                items = sorted(os.listdir(path))
                for i, item in enumerate(items):
                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1
                    
                    new_prefix = indent * depth
                    if is_last:
                        new_prefix += '└── '
                    else:
                        new_prefix += '├── '
                    
                    if os.path.isfile(item_path):
                        if not should_ignore(item_path, start_path, ignore_patterns, False):
                            tree.append(f"{new_prefix}{item}")
                    elif os.path.isdir(item_path):
                        if not should_ignore(item_path, start_path, ignore_patterns, True):
                            tree.append(f"{new_prefix}{item}")
                            build_tree(item_path, depth + 1)
            except PermissionError:
                tree.append(f"{indent * (depth + 1)}[Permission Denied]")
    
    build_tree(start_path)
    
    result = '\n'.join(tree)
    if file_output:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"目录架构已保存到 {output_file}")
    else:
        print(result)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成文件夹架构树状图（支持.gitignore）')
    parser.add_argument('-d', '--depth', type=int, help='最大遍历深度')
    parser.add_argument('-o', '--output', action='store_true', help='输出到文件')
    parser.add_argument('-f', '--file', default='directory_tree.txt', help='输出文件名')
    parser.add_argument('--no-gitignore', action='store_false', dest='respect_gitignore', 
                       help='不遵守.gitignore规则')
    parser.set_defaults(respect_gitignore=True)
    args = parser.parse_args()
    
    generate_directory_tree(
        max_depth=args.depth,
        file_output=args.output,
        output_file=args.file,
        respect_gitignore=args.respect_gitignore
    )