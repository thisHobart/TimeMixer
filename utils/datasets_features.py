from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. 自动定位项目路径
# =========================
# 当前文件位置: 项目根目录 / utils / build_power_features.py
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

INPUT_PATH = PROJECT_ROOT / "dataset" / "datasets.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "datasets_power_features.csv"


# =========================
# 2. 工具函数
# =========================
def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    读取 CSV，兼容 utf-8-sig / gbk。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到原始数据文件: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    安全除法，避免除以 0。
    """
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(0)


def check_required_columns(df: pd.DataFrame, required_cols: list):
    """
    检查必要字段是否存在。
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"缺少必要字段: {missing_cols}\n"
            f"当前数据字段为: {list(df.columns)}"
        )


# =========================
# 3. 构造预测侧特征
# =========================
def add_forecast_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造预测侧火电、水电结构特征。

    原始数据中没有 forecast_火电，需要由以下字段计算：

        forecast_火电 = forecast_总和 - forecast_新能源 - forecast_水电 - forecast_非市场机组

    需要字段:
        forecast_负荷
        forecast_总和
        forecast_新能源
        forecast_水电
        forecast_非市场机组

    生成字段:
        forecast_火电
        forecast_水电_ratio
        forecast_火电_ratio
        forecast_净负荷
        forecast_火电_minus_水电
        forecast_水电_diff
        forecast_火电_diff
    """

    required_cols = [
        "forecast_负荷",
        "forecast_总和",
        "forecast_新能源",
        "forecast_水电",
        "forecast_非市场机组",
    ]

    check_required_columns(df, required_cols)

    df = df.copy()

    # 1. 计算预测火电
    df["forecast_火电"] = (
        df["forecast_总和"]
        - df["forecast_新能源"]
        - df["forecast_水电"]
        - df["forecast_非市场机组"]
    )

    # 2. 预测水电 / 预测负荷
    df["forecast_水电_ratio"] = safe_divide(
        df["forecast_水电"],
        df["forecast_负荷"],
    )

    # 3. 预测火电 / 预测负荷
    df["forecast_火电_ratio"] = safe_divide(
        df["forecast_火电"],
        df["forecast_负荷"],
    )

    # 4. 预测负荷 - 预测水电
    df["forecast_净负荷"] = df["forecast_负荷"] - df["forecast_水电"]

    # 5. 预测火电 - 预测水电
    df["forecast_火电_minus_水电"] = df["forecast_火电"] - df["forecast_水电"]

    # 6. 预测水电变化量
    df["forecast_水电_diff"] = df["forecast_水电"].diff().fillna(0)

    # 7. 预测火电变化量
    df["forecast_火电_diff"] = df["forecast_火电"].diff().fillna(0)

    return df

# =========================
# 4. 构造实际侧特征
# =========================
def add_actual_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造实际侧火电、水电结构特征。

    需要字段:
        load
        quantity_火电
        quantity_水电

    生成字段:
        actual_水电_ratio
        actual_火电_ratio
        actual_净负荷
        actual_火电_minus_水电
        actual_水电_diff
        actual_火电_diff
    """

    required_cols = [
        "load",
        "quantity_火电",
        "quantity_水电",
    ]

    check_required_columns(df, required_cols)

    df = df.copy()

    # 实际水电 / 实际负荷
    df["actual_水电_ratio"] = safe_divide(
        df["quantity_水电"],
        df["load"],
    )

    # 实际火电 / 实际负荷
    df["actual_火电_ratio"] = safe_divide(
        df["quantity_火电"],
        df["load"],
    )

    # 实际负荷 - 实际水电
    df["actual_净负荷"] = df["load"] - df["quantity_水电"]

    # 实际火电 - 实际水电
    df["actual_火电_minus_水电"] = df["quantity_火电"] - df["quantity_水电"]

    # 实际水电变化量
    df["actual_水电_diff"] = df["quantity_水电"].diff().fillna(0)

    # 实际火电变化量
    df["actual_火电_diff"] = df["quantity_火电"].diff().fillna(0)

    return df


# =========================
# 5. 构造时间特征
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造基础时间特征。
    需要字段:
        datetime
    """

    if "datetime" not in df.columns:
        print("未检测到 datetime 字段，跳过时间特征构造。")
        return df

    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # 小时周期编码
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 星期周期编码
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # 月份周期编码
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 是否周末
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # 四川水电相关季节特征，可后续按业务规则调整
    df["is_wet_season"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    df["is_dry_season"] = df["month"].isin([12, 1, 2, 3]).astype(int)

    return df


# =========================
# 6. 清理异常值
# =========================
def clean_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理新特征中的 inf / nan。
    """

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        col for col in df.columns
        if (
            "ratio" in col
            or "diff" in col
            or "净负荷" in col
            or "minus" in col
            or col.endswith("_sin")
            or col.endswith("_cos")
            or col.startswith("is_")
        )
    ]

    df[feature_cols] = df[feature_cols].fillna(0)

    return df


# =========================
# 7. 主处理流程
# =========================
def build_power_features():
    print("=" * 60)
    print("开始构造四川电价预测特征")
    print("=" * 60)

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")

    df = read_csv_safely(INPUT_PATH)

    print(f"\n原始数据维度: {df.shape}")
    print(f"原始字段: {list(df.columns)}")

    df = add_time_features(df)
    df = add_forecast_power_features(df)
    df = add_actual_power_features(df)
    df = clean_feature_values(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n处理完成")
    print(f"输出数据维度: {df.shape}")
    print(f"已保存到: {OUTPUT_PATH}")

    print("\n预测侧新增特征:")
    forecast_feature_cols = [
        "forecast_火电",
        "forecast_水电_ratio",
        "forecast_火电_ratio",
        "forecast_净负荷",
        "forecast_火电_minus_水电",
        "forecast_水电_diff",
        "forecast_火电_diff",
    ]

    for col in forecast_feature_cols:
        print(f"  - {col}")

    print("\n实际侧新增特征:")
    actual_feature_cols = [
        "actual_水电_ratio",
        "actual_火电_ratio",
        "actual_净负荷",
        "actual_火电_minus_水电",
        "actual_水电_diff",
        "actual_火电_diff",
    ]

    for col in actual_feature_cols:
        print(f"  - {col}")

    print("=" * 60)


if __name__ == "__main__":
    build_power_features()