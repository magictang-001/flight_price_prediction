import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
BASE_ROOT = r"d:\课堂\大四\Python\Section 5"
CSV_PATH = os.path.join(
    BASE_ROOT,
    "1. Flight Fare Prediction Project-1 Predicting and Analyzing Flight Ticket Prices",
    "机票数据.csv",
)
def read_csv_with_encoding(path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)
def parse_time(val):
    try:
        t = pd.to_datetime(val, errors="coerce")
        if pd.isna(t):
            raise ValueError("coerce")
        return int(t.hour), int(t.minute)
    except Exception:
        try:
            import re
            s = str(val).strip()
            m = re.search(r'(\d{1,2})\\s*[:：]\\s*(\d{1,2})', s)
            if m:
                h = int(m.group(1))
                mn = int(m.group(2))
                return h, mn
            parts = s.split(":")
            h = int(re.sub(r'\\D+', '', parts[0])) if parts and parts[0] else np.nan
            mn = int(re.sub(r'\\D+', '', parts[1])) if len(parts) > 1 and parts[1] else 0
            if pd.isna(h):
                return np.nan, np.nan
            return h, mn
        except Exception:
            return np.nan, np.nan
def parse_date(val):
    try:
        s = str(val).strip()
        if s == "":
            return np.nan, np.nan
        import re
        m = re.search(r'(\d{1,2})月(\d{1,2})日', s)
        if m:
            mm = int(m.group(1))
            dd = int(m.group(2))
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return dd, mm
        m2 = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', s)
        if m2:
            yy = int(m2.group(1))
            mm = int(m2.group(2))
            dd = int(m2.group(3))
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return dd, mm
        m3 = re.search(r'(\d{1,2})[\\-/](\d{1,2})', s)
        if m3:
            mm = int(m3.group(1))
            dd = int(m3.group(2))
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return dd, mm
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(d):
            return np.nan, np.nan
        return int(d.day), int(d.month)
    except Exception:
        return np.nan, np.nan
def build_features(df):
    df.columns = df.columns.str.strip()
    col_date = "出发日期" if "出发日期" in df.columns else ("Date_of_Journey" if "Date_of_Journey" in df.columns else None)
    col_dep = "出发时间" if "出发时间" in df.columns else ("Dep_Time" if "Dep_Time" in df.columns else None)
    col_arr = "抵达时间" if "抵达时间" in df.columns else ("Arrival_Time" if "Arrival_Time" in df.columns else None)
    col_price = "机票价格" if "机票价格" in df.columns else ("Price" if "Price" in df.columns else None)
    col_airline = "航空公司" if "航空公司" in df.columns else ("Airline" if "Airline" in df.columns else None)
    col_source = "出发城市" if "出发城市" in df.columns else ("Source" if "Source" in df.columns else None)
    col_dest = "到达城市" if "到达城市" in df.columns else ("Destination" if "Destination" in df.columns else None)
    col_model = "客机机型" if "客机机型" in df.columns else ("Aircraft_Model" if "Aircraft_Model" in df.columns else None)
    if col_price is None:
        raise RuntimeError("CSV缺少价格列：机票价格")
    # date parts
    if col_date:
        dm = df[col_date].apply(parse_date)
        journey_day = np.array([x[0] for x in dm])
        journey_month = np.array([x[1] for x in dm])
    else:
        journey_day = np.full(len(df), np.nan)
        journey_month = np.full(len(df), np.nan)
    # times
    if col_dep:
        dep = df[col_dep].apply(parse_time)
        dep_hour = np.array([x[0] for x in dep])
        dep_min = np.array([x[1] for x in dep])
    else:
        dep_hour = np.full(len(df), np.nan)
        dep_min = np.full(len(df), np.nan)
    if col_arr:
        arr = df[col_arr].apply(parse_time)
        arr_hour = np.array([x[0] for x in arr])
        arr_min = np.array([x[1] for x in arr])
    else:
        arr_hour = np.full(len(df), np.nan)
        arr_min = np.full(len(df), np.nan)
    # duration, assume same day; wrap negatives by +24h
    dur_hour = arr_hour - dep_hour
    dur_min = arr_min - dep_min
    mask_neg = (dur_hour < 0) | ((dur_hour == 0) & (dur_min < 0))
    dur_hour[mask_neg] = dur_hour[mask_neg] + 24
    X = pd.DataFrame({
        "Journey_day": journey_day,
        "Journey_month": journey_month,
        "Dep_hour": dep_hour,
        "Dep_min": dep_min,
        "Arrival_hour": arr_hour,
        "Arrival_min": arr_min,
        "Duration_hours": dur_hour,
        "Duration_mins": dur_min,
        "Airline": df[col_airline] if col_airline else pd.Series("", index=df.index),
        "Source": df[col_source] if col_source else pd.Series("", index=df.index),
        "Destination": df[col_dest] if col_dest else pd.Series("", index=df.index),
        "Aircraft_Model": df[col_model] if col_model else pd.Series("", index=df.index),
    })
    cleaned_price = df[col_price].astype(str).str.replace(r'[^0-9\\.]', '', regex=True)
    y = pd.to_numeric(cleaned_price, errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]
def train_and_save():
    df = read_csv_with_encoding(CSV_PATH)
    X, y = build_features(df)
    if len(X) == 0:
        raise RuntimeError("数据特征为空，无法训练模型。请检查CSV的日期与时间格式。")
    if len(X) < 50:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_cols = ["Journey_day","Journey_month","Dep_hour","Dep_min","Arrival_hour","Arrival_min","Duration_hours","Duration_mins"]
    cat_cols = ["Airline","Source","Destination","Aircraft_Model"]
    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    model = LinearRegression()
    pipe = Pipeline(steps=[("preproc", preproc), ("model", model)])
    print("Fitting model...")
    pipe.fit(X_train, y_train)
    web_dir = os.path.join(BASE_ROOT, "2. Deploying the Flight Fare Prediction Model with Flask Framework Making Prediction")
    out_path = os.path.join(web_dir, "flight_xgb.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print("Model saved to:", out_path)
    print("Train size:", len(X_train), "Test size:", len(X_test))
if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as e:
        import traceback
        print("Training failed:", str(e))
        traceback.print_exc()
