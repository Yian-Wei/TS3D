import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# 1. 定义日志模板数据
# 给定的模板字符串
template_str = """EventId,EventTemplate
1,.*Database connection successful.*
2,.*Order data query successful.*
3,.*Order update successful.*
4,.*Order deletion successful.*
5,.*Order insertion successful.*
6,.*Order data query failed.*
7,.*Order update failed.*
8,.*Order deletion failed.*
9,.*Order insertion failed.*
10,.*Payment data query successful.*
11,.*Payment deletion successful.*
12,.*Payment insertion successful.*
13,.*Payment update successful.*
14,.*Payment data query failed.*
15,.*Payment deletion failed.*
16,.*Payment insertion failed.*
17,.*Payment update failed.*
18,.*User data query successful.*
19,.*User update successful.*
20,.*User deletion successful.*
21,.*User insertion successful.*
22,.*User data query failed.*
23,.*User update failed.*
24,.*User deletion failed.*
25,.*User insertion failed.*
26,.*Driver data query successful.*
27,.*Driver insertion successful.*
28,.*Driver deletion successful.*
29,.*Driver update successful.*
30,.*Driver data query failed.*
31,.*Driver insertion failed.*
32,.*Driver deletion failed.*
33,.*Driver update failed.*
34,.*floating point alignment exceptions.*
35,.*select.* query successful.*
36,.*select.* query failed.*
37,.*update.* execution failed.*
38,.*update.* execution successful.*
39,.*Task .*: API call successful.*
40,.*Task .*: API call failed.*
41,.*:.* .*:.*"""

# 解析模板字符串的代码
log_templates = []
lines = template_str.strip().split('\n')[1:]  # 跳过标题行

for line in lines:
    parts = line.split(',', 1)  # 只分割一次，保留EventTemplate中的逗号
    if len(parts) == 2:
        event_id, event_template = parts
        log_templates.append({
            "EventId": event_id.strip(),
            "EventTemplate": event_template.strip()
        })

# 2. 保存为JSON文件
def save_templates_to_json(templates, file_path):
    """将模板数据保存为JSON文件"""
    # 转换为字典格式：{EventId: {模板数据}}
    template_dict = {
        t["EventId"]: {"EventTemplate": t["EventTemplate"]} for t in templates
    }

    with open(file_path, "w") as f:
        json.dump(template_dict, f, indent=2)
    print(f"✅ 模板已保存至: {file_path}")


# 3. 从JSON文件加载模板并转向量
def templates_to_vectors(json_file_path):
    """将JSON文件中的模板转换为向量表示"""
    # 加载模板数据
    with open(json_file_path, "r") as f:
        template_data = json.load(f)

    # 提取模板文本
    templates = [data["EventTemplate"] for data in template_data.values()]
    event_ids = list(template_data.keys())

    # 预处理模板：移除正则符号，保留关键词语
    processed_templates = []
    for template in templates:
        # 移除正则特殊字符.* 保留关键词语
        cleaned = template.replace(".*", " ").replace(":", " ").strip()
        # 移除多余空格
        cleaned = " ".join(cleaned.split())
        processed_templates.append(cleaned)

    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(
        token_pattern=r"\b\w+\b",  # 单词级token
        max_features=100,  # 保留前100个重要特征
        stop_words="english",  # 移除英文停用词
    )

    # 生成向量
    vectors = vectorizer.fit_transform(processed_templates)

    # 归一化处理（可选）
    normalized_vectors = normalize(vectors, norm="l2")

    # 创建向量字典
    vector_dict = {}
    for idx, event_id in enumerate(event_ids):
        vector_dict[event_id] = normalized_vectors[idx].toarray()[0].tolist()

    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()

    print(f"🔢 成功生成 {len(vector_dict)} 个模板向量")
    print(f"📐 向量维度: {vectors.shape[1]}")
    return vector_dict, feature_names


# 4. 保存向量到JSON文件
def save_vectors_to_json(vector_dict, file_path):
    """将向量字典保存为JSON文件"""
    with open(file_path, "w") as f:
        json.dump(vector_dict, f, indent=2)
    print(f"💾 向量已保存至: {file_path}")


# 5. 主程序
if __name__ == "__main__":
    # 保存模板到JSON
    save_templates_to_json(log_templates, "log_templates.json")

    # 加载模板并转向量
    vector_dict, feature_names = templates_to_vectors("log_templates.json")

    # 保存向量到JSON
    save_vectors_to_json(vector_dict, "template_vectors.json")
