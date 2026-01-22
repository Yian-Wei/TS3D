import pandas as pd
import re
from datetime import datetime
import glob

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

print("加载 metrics ...")
metrics_files = glob.glob("m2m/metrics/mysql-*_metrics.txt")
metrics_list = []

for f in metrics_files:
    df = pd.read_csv(f)
    if "label" not in df.columns:
        raise ValueError(f"文件 {f} 缺少 'label' 列，请检查格式。")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["label"] = df["label"].astype(int)  
    metrics_list.append(df)

metrics = pd.concat(metrics_list, ignore_index=True)
metrics = metrics.sort_values(["label", "timestamp"])

print("解析日志 ...")
time_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
addr_pattern = re.compile(r"\[(\d+\.\d+\.\d+\.\d+:\d+)\]")

log_lines = []
with open("m2m/logs/logs.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        time_match = time_pattern.search(line)
        if not time_match:
            continue
        ts = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
        addr_match = addr_pattern.search(line)
        addr = addr_match.group(1) if addr_match else None
        label = ip_to_label.get(addr)
        if label is None:
            continue
        log_lines.append({"timestamp": ts, "addr": addr, "label": label, "raw_log": line})

logs = pd.DataFrame(log_lines)
logs = logs.sort_values(["label", "timestamp"])
logs["label"] = logs["label"].astype(int)

print("执行快速时间匹配合并（merge_asof） ...")


logs = logs.sort_values("timestamp").reset_index(drop=True)
metrics = metrics.sort_values("timestamp").reset_index(drop=True)

merged = pd.merge_asof(
    logs,
    metrics,
    on="timestamp",
    direction="nearest",                  
    tolerance=pd.Timedelta("1s")          
)

print("保存结果 ...")

cols_front = ["timestamp", "addr", "raw_log"]
other_cols = [c for c in merged.columns if c not in cols_front]
merged = merged[cols_front + other_cols]

merged.to_csv("m2m/merged_output.csv", index=False, encoding="utf-8")
print("✅ 合并完成，结果已保存到 merged_output.csv")