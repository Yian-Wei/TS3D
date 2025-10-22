import pandas as pd
import numpy as np

def has_nan_scalar(d):
    for v in d.values():
        if isinstance(v, (list, np.ndarray)):
            continue
        if pd.isna(v):
            return True
    return False

def aggregate_window(df, eps=1e-5):
    drop_cols = ["operation_type", "select_star_flag", "join_flag", "label_x"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.sort_values(by=["ip_label", "timestamp"]).reset_index(drop=True)

    num_cols = df.columns.tolist()
    start_idx = num_cols.index("cpu")
    value_cols = num_cols[start_idx:]  

    aggregated_rows = []
    curr_window = []
    prev_values = None

    for _, row in df.iterrows():
        curr_values = row[value_cols].to_numpy(dtype=float)

        if prev_values is None or not np.all(np.abs(curr_values - prev_values) < eps):
            if curr_window:
                agg_row = {}
                agg_row["start_time"] = curr_window[0]["timestamp"]
                agg_row["metric_source_ip"] = curr_window[0]["ip_label"]
                agg_row["num_events"] = len(curr_window)
                agg_row["aggregated_log_level"] = int(any(r["log_level"] for r in curr_window))
                agg_row["max_response_time"] = max(r["response_time"] for r in curr_window)
                agg_row["event_ids_list"] = [r["eventid"] for r in curr_window]
                for c in value_cols:
                    agg_row[c] = curr_window[0][c]
                agg_row["label"] = int(any(r["label"] for r in curr_window))

                if not has_nan_scalar(agg_row):
                    aggregated_rows.append(agg_row)

            curr_window = [row]
            prev_values = curr_values
        else:
            curr_window.append(row)

    if curr_window:
        agg_row = {}
        agg_row["start_time"] = curr_window[0]["timestamp"]
        agg_row["metric_source_ip"] = curr_window[0]["ip_label"]
        agg_row["num_events"] = len(curr_window)
        agg_row["aggregated_log_level"] = int(any(r["log_level"] for r in curr_window))
        agg_row["max_response_time"] = max(r["response_time"] for r in curr_window)
        agg_row["event_ids_list"] = [r["eventid"] for r in curr_window]
        for c in value_cols:
            agg_row[c] = curr_window[0][c]
        agg_row["label"] = int(any(r["label"] for r in curr_window))

        if not has_nan_scalar(agg_row):
            aggregated_rows.append(agg_row)

    agg_df = pd.DataFrame(aggregated_rows)

    agg_df = agg_df.sort_values(by="start_time").reset_index(drop=True)
    return agg_df

if __name__ == "__main__":
    df = pd.read_csv("m2m/final_output.csv")  
    agg_df = aggregate_window(df)
    agg_df.to_csv("m2m/aggregated_output.csv", index=False)
    print("✅ 聚合完成，结果已保存到 aggregated_output.csv")