import csv
import ast
import re


def clean_sequence(seq):
    """对于数字列表，直接转换为字符串并用空格连接"""
    return " ".join(str(item) for item in seq)


csv.field_size_limit(10485760)
# 使用流式处理，避免一次性加载整个数据集
with open("data/ts/test.csv", "r", newline="") as csvfile, open(
    "data/ts/s2s_test_normal", "w"
) as f0, open("data/ts/s2s_test_abnormal", "w") as f1:

    reader = csv.DictReader(csvfile)

    for i, row in enumerate(reader):
        # 每处理10000行显示一次进度
        if i % 10000 == 0 and i > 0:
            print(f"已处理 {i} 行...")

        try:
            # 高效解析序列
            sequence_str = row["sequence"].strip()
            if sequence_str.startswith("["):
                sequence = ast.literal_eval(sequence_str)
            else:
                # 处理可能的格式变化
                sequence = sequence_str[1:-1].replace("'", "").split(", ")

            # 清理序列
            cleaned_seq = clean_sequence(sequence)
            if len(sequence)== 0:
                continue
            # 根据标签写入相应文件
            if row["label"] == "0":
                f0.write(cleaned_seq + "\n")
            elif row["label"] == "1":
                f1.write(cleaned_seq + "\n")

        except Exception as e:
            print(f"处理第 {i} 行时出错: {e}")
            continue

print("处理完成！结果已保存到 label0.txt 和 label1.txt")
