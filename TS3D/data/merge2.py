import pandas as pd
import re

ip_map = {
    0: "10.244.0.16:3306",
    1: "10.244.0.18:3306",
    2: "10.244.0.20:3306",
    3: "10.244.0.22:3306",
    4: "10.244.0.24:3306",
    5: "10.244.0.26:3306",
    6: "10.244.0.28:3306",
}
extra_addrs = ["172.18.0.2:30007"]
ip_to_label = {v: k for k, v in ip_map.items()}
for addr in extra_addrs:
    ip_to_label[addr] = 0

event_templates = {
    1: ".*Database connection successful.*",
    2: ".*Database connection failed.*",
    3: ".*Database connection not initialized.*",

    4: r"^\[.*?\]\[SELECT\s+(?!\*\b)(?!.*\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\()[^*]+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s*(?![^\]]*(?:\(|\bWHERE\b|\bJOIN\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bEXCEPT\b|\bINTERSECT\b))\s*;?\](?:\[.*?\])*?$",
    5: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+WHERE\s+(?![^]]*\bEXISTS\b)(?![^]]*\bFOR\s+UPDATE\b).+?;?\](?:\[.*?\])*?$",

    6: r"^\[.*?\]\[SELECT\s+\*\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s*;?\](?:\[.*?\])*?$",  
    7: r"^\[.*?\]\[SELECT\s+\*\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+LIMIT\s+\d+\s*;?\](?:\[.*?\])*?$",  

    8: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+\S+(?:\s+\w+)?\s+WHERE\s+EXISTS\s*\(\s*SELECT\s+.+?\s+FROM\s+\S+(?:\s+\w+)?(?:\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?)+\s+WHERE\s+.+?\)\s*;?\](?:\[.*?\])*?$",
    9: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?(?:\s+JOIN\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+ON\s+.+?){2,}\s+WHERE\s+(?![^]]*\(\s*SELECT\b).+?;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",
    10: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+\S+(?:\s+\w+)?\s+LEFT\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?(?:\s+LEFT\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?)+\s+ORDER\s+BY\s+.+?;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",
    11: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+\S+(?:\s+\w+)?(?:\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?)+\s+WHERE\s+.+?\(\s*SELECT\s+.+?\s+FROM\s+\S+.*?\).*?;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",

    12: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?(?:\s+JOIN\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+ON\s+.+?){2,}\s+WHERE\s+(?![^]]*\(\s*SELECT\b).+?;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",
    13: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+\S+(?:\s+\w+)?\s+LEFT\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?(?:\s+LEFT\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?)+\s+ORDER\s+BY\s+.+?;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",
    14: r"^\[.*?\]\[SELECT\s+.+?\s+FROM\s+\S+(?:\s+\w+)?(?:\s+JOIN\s+\S+(?:\s+\w+)?\s+ON\s+.+?)+\s+WHERE\s+.+?\(\s*SELECT\s+.+?\s+FROM\s+\S+.*?\).*?;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",

    15: r"^\[.*?\]\[SELECT\s+.*?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\([^)]*\).*?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s*(?![^\]]*\bJOIN\b)(?![^\]]*\bGROUP\s+BY\b)\s*;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",
    16: r"^\[.*?\]\[SELECT\s+.+?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\(.+?\).+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s*(?![^\]]*\bJOIN\b)\s+GROUP\s+BY\s+.+?;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",
    17: r"^\[.*?\]\[(?![^\]]*\bGROUP\s+BY\b)SELECT\s+.+?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\(.+?\).+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?(?:\s+JOIN\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+ON\s+.+?)+\s*;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",
    18: r"^\[.*?\]\[SELECT\s+.+?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\(.+?\).+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?(?:\s+JOIN\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+ON\s+.+?)+\s+GROUP\s+BY\s+.+?;?\](?:\[.*?\])*\[response time:(?:0\.(?:[4-9]\d*|3\d*[1-9]\d*)|[1-9]\d*(?:\.\d+)?)\]$",

    19: r"^\[.*?\]\[SELECT\s+.*?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\([^)]*\).*?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s*(?![^\]]*\bJOIN\b)(?![^\]]*\bGROUP\s+BY\b)\s*;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",
    20: r"^\[.*?\]\[SELECT\s+.+?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\(.+?\).+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s*(?![^\]]*\bJOIN\b)\s+GROUP\s+BY\s+.+?;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",
    21: r"^\[.*?\]\[(?![^\]]*\bGROUP\s+BY\b)SELECT\s+.+?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\(.+?\).+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?(?:\s+JOIN\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+ON\s+.+?)+\s*;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",
    22: r"^\[.*?\]\[SELECT\s+.+?\b(?:AVG|SUM|MAX|MIN|COUNT)\s*\(.+?\).+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?(?:\s+JOIN\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+ON\s+.+?)+\s+GROUP\s+BY\s+.+?;?\](?:\[.*?\])*\[response time:0\.[0-2]\d*\]$",

    23: r"^\[.*?\]\[UPDATE\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+SET\s+.+?\s+WHERE\s+.+?;?\](?:\[.*?\])*?$",
    24: r"^[^\[]*\[.*?\]\[SELECT\s+.+?\s+FROM\s+[A-Za-z0-9_`\.]+(?:\s+\w+)?\s+WHERE\s+.+?\s+FOR\s+UPDATE\s*;?\](?:\[.*?\])*?$",
    25: r"^\[.*?\]\[SELECT\s+\w+\s*\([^)]*\)\s*;?\](?:\[.*?\])*?$",

    26: r"^\[.*?\]\[INSERT\s+INTO\s+\S+(?:\s+\w+)?\s*\(.+?\)\s+VALUES\s*\(.+?\)\s*;?\](?:\[.*?\])*?$",
    27: ".*Cannot add or update a child row.*",
    28: ".*Duplicate entry.*",

    29: ".*Task\s*\d+:\s*API call successful.*",
    30: ".*Task\s*\d+:\s*API call failed.*",

    31: r"^\[.*?\]\[DELETE\s+FROM\s+\S+(?:\s+\w+)?\s+WHERE\s+.+?;?\](?:\[.*?\])*?$",

    32: ".*:.* .*:.*",
}


def extract_info(raw_log):
    parts = re.findall(r"\[(.*?)\]", raw_log)
    if len(parts) < 3:
        return None, None, None, None

    ts = parts[0]     # timestamp
    addr = parts[1]   # ip:port

    level_idx = None
    for i, p in enumerate(parts):
        if p in ("info", "error", "warning"):
            level_idx = i
            break

    if level_idx is None or level_idx == len(parts) - 1:
        return ts, addr, None, None

    level = parts[level_idx]
    message = parts[level_idx + 1]
    return ts, addr, level, message


def extract_operation_type(raw_log: str) -> int:

    if pd.isna(raw_log):
        return 4
    text = str(raw_log).lower()
    if "connection" in text:
        return 0
    elif "update" in text:
        return 1
    elif "deletion" in text:
        return 2
    elif "insertion" in text:
        return 3
    else:
        return 4


def extract_select_star(raw_log: str) -> int:

    if pd.isna(raw_log):
        return 0
    return 1 if re.search(r"select\s*\*", str(raw_log), re.IGNORECASE) else 0


def extract_join_flag(raw_log: str) -> int:

    if pd.isna(raw_log):
        return 0
    return 1 if re.search(r"\bjoin\b", str(raw_log), re.IGNORECASE) else 0


def map_eventid_fast(raw_log):
    if pd.isna(raw_log):
        return None
    text = str(raw_log).lower()
    
    compiled_templates = [(eid, re.compile(pattern, re.IGNORECASE)) for eid, pattern in event_templates.items()]

    if "select" in text:
        pats = [p for p in compiled_templates if "select" in p[1].pattern.lower()]
    elif "update" in text:
        pats = [p for p in compiled_templates if "update" in p[1].pattern.lower()]
    elif "insert" in text:
        pats = [p for p in compiled_templates if "insert" in p[1].pattern.lower()]
    elif "connection" in text:
        pats = [p for p in compiled_templates if "connection" in p[1].pattern.lower()]
    else:
        pats = compiled_templates
    for eid, pat in pats:
        if pat.search(text):
            return eid
    return None
def map_log_level(level):
    if level == "info":
        return 0
    elif level == "error":
        return 1
    return 2

def extract_response_time(raw_log):

    if pd.isna(raw_log):
        return 0.0
    match = re.search(r"response time:([\d\.]+)", str(raw_log))
    if match:
        return float(match.group(1))
    return 0.0


def detect_abnormal_label(raw_log, response_time):

    if pd.isna(raw_log):
        return 0
    text = str(raw_log).lower()
    if response_time > 0.3 or "error" in text:
        return 1
    return 0


df = pd.read_csv("m2m/merged_output.csv")
if "label_y" in df.columns:
    df = df.drop(columns=["label_y"])

df[["timestamp_raw", "addr", "log_level_raw", "log_message"]] = df["raw_log"].apply(
    lambda x: pd.Series(extract_info(x))
)

df["response_time"] = df["raw_log"].apply(extract_response_time)
df["eventid"] = df["raw_log"].apply(map_eventid_fast)
df["log_level"] = df["log_level_raw"].apply(map_log_level)
df["ip_label"] = df["addr"].map(ip_to_label)
df["operation_type"] = df["raw_log"].apply(extract_operation_type)
df["select_star_flag"] = df["raw_log"].apply(extract_select_star)
df["join_flag"] = df["raw_log"].apply(extract_join_flag)



df["label"] = df.apply(lambda row: detect_abnormal_label(row["raw_log"], row["response_time"]), axis=1)

df["timestamp"] = df["timestamp_raw"]

df = df.drop(columns=["timestamp_raw", "raw_log", "log_level_raw", "log_message", "addr"])

first_cols = [
    "label", "ip_label", "timestamp", "eventid", "log_level",
    "response_time", "operation_type",
    "select_star_flag", "join_flag", 
]
other_cols = [c for c in df.columns if c not in first_cols]
df = df[first_cols + other_cols]

total = len(df)
abnormal = (df["label"] == 1).sum()
normal = total - abnormal
ratio = abnormal / total * 100 if total > 0 else 0

df.to_csv("m2m/final_output.csv", index=False, encoding="utf-8")

print("✅ 处理完成，结果已保存到 final_output.csv")
print(f"📊 日志总数: {total}")
print(f"   正常样本数: {normal}")
print(f"   异常样本数: {abnormal}")
print(f"   异常比例: {ratio:.2f}%")