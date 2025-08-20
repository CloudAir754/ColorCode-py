# 输出颜色矩阵和人名的映射关系

from pprint import pprint

# 与题目完全一致的常量
KEY_POS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
FIRST   = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank"]
LAST    = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]

# 预生成 64 个名字
NAMES = [f"{FIRST[i % 8]} {LAST[i // 8]}" for i in range(64)]

def bin_to_matrix(bits: str):
    """
    把 6 位二进制字符串转成 3×3 颜色矩阵
    其余位置填 'Black' 占位
    """
    # 先全部填 Green
    mat = [['Green']*3 for _ in range(3)]
    for idx, (r, c) in enumerate(KEY_POS):
        mat[r][c] = 'Red' if bits[idx] == '1' else 'Blue'
    return mat

def main():
    mapping = {}
    for num in range(64):
        bits = f"{num:06b}"            # 0-padded 6-bit
        matrix = bin_to_matrix(bits)   # 3×3 颜色矩阵
        name = NAMES[num]              # 对应人名
        mapping[bits] = {"matrix": matrix, "name": name}

    # 打印结果（64 组）
    for bits, item in mapping.items():
        print(f"{bits} -> {item['name']}")
        pprint(item["matrix"], width=40)
        print("-" * 30)

if __name__ == "__main__":
    main()