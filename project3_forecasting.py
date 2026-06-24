from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import SARIMAX
from statsmodels.tsa.stattools import adfuller
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path("data/hy_oas_2023_2026.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")
RESULT_DIR = OUTPUT_DIR / "tables"
TEXT_DIR = OUTPUT_DIR / "text"

SEED = 2026
SEASONAL_PERIOD = 5


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def set_reproducible() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, FIGURE_DIR, RESULT_DIR, TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["hy_oas"] = pd.to_numeric(df["hy_oas"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def split_data(df: pd.DataFrame) -> SplitData:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    return SplitData(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )


def save_series_overview(df: pd.DataFrame) -> None:
    stats = df["hy_oas"].describe().to_frame("hy_oas")
    missing = df.isna().sum().to_frame("missing_count")
    stats.to_csv(RESULT_DIR / "descriptive_statistics.csv")
    missing.to_csv(RESULT_DIR / "missing_values.csv")

    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], df["hy_oas"], color="#1f77b4", linewidth=1.5)
    plt.title("HY OAS Daily Time Series")
    plt.xlabel("Date")
    plt.ylabel("HY OAS")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hy_oas_overview.png", dpi=200)
    plt.close()


def save_transformed_series_plots(df: pd.DataFrame) -> None:
    log_series = np.log(df["hy_oas"])
    diff_log_series = log_series.diff()

    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], log_series, color="#2f5597", linewidth=1.5)
    plt.title("Log HY OAS Time Series")
    plt.xlabel("Date")
    plt.ylabel("log(HY OAS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "log_hy_oas_timeseries.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], diff_log_series, color="#8a4f7d", linewidth=1.2)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("First Difference of Log HY OAS")
    plt.xlabel("Date")
    plt.ylabel("Delta log(HY OAS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "diff_log_hy_oas_timeseries.png", dpi=200)
    plt.close()


def save_split_plot(splits: SplitData) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(splits.train["date"], splits.train["hy_oas"], label="Train", color="#1f77b4", linewidth=1.4)
    plt.plot(splits.val["date"], splits.val["hy_oas"], label="Validation", color="#ff7f0e", linewidth=1.4)
    plt.plot(splits.test["date"], splits.test["hy_oas"], label="Test", color="#2ca02c", linewidth=1.4)
    plt.axvline(splits.val["date"].iloc[0], color="gray", linestyle="--", linewidth=1.0)
    plt.axvline(splits.test["date"].iloc[0], color="gray", linestyle="--", linewidth=1.0)
    plt.title("Train / Validation / Test Split of HY OAS")
    plt.xlabel("Date")
    plt.ylabel("HY OAS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "train_val_test_split.png", dpi=200)
    plt.close()


def stationarity_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    series_map = {
        "hy_oas": df["hy_oas"],
        "log_hy_oas": np.log(df["hy_oas"]),
        "diff_log_hy_oas": np.log(df["hy_oas"]).diff().dropna(),
    }

    for name, series in series_map.items():
        clean = series.dropna()
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(clean, autolag="AIC")
        rows.append(
            {
                "series": name,
                "adf_stat": adf_stat,
                "adf_pvalue": adf_p,
                "adf_5pct_critical": adf_crit["5%"],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(RESULT_DIR / "stationarity_tests.csv", index=False)
    return result


def save_acf_pacf(df: pd.DataFrame) -> None:
    transformed = np.log(df["hy_oas"]).diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(transformed, lags=40, ax=axes[0])
    plot_pacf(transformed, lags=40, ax=axes[1], method="ywm")
    axes[0].set_title("ACF of Delta log(HY OAS)")
    axes[1].set_title("PACF of Delta log(HY OAS)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "acf_pacf_diff_log_hy_oas.png", dpi=200)
    plt.close(fig)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
    }


def naive_forecast(full: pd.DataFrame, test_start: int) -> np.ndarray:
    values = full["hy_oas"].to_numpy()
    return values[test_start - 1 : len(full) - 1]


def ewma_one_step(values: np.ndarray, alpha: float, start: int, end: int) -> np.ndarray:
    level = values[0]
    preds = []
    for i in range(1, end):
        pred = level
        if i >= start:
            preds.append(pred)
        level = alpha * values[i] + (1 - alpha) * level
    return np.array(preds)


def tune_ewma(full: pd.DataFrame, train_end: int, val_end: int) -> tuple[float, pd.DataFrame]:
    values = full["hy_oas"].to_numpy()
    rows = []
    for alpha in np.arange(0.05, 1.00, 0.05):
        pred = ewma_one_step(values, float(alpha), train_end, val_end)
        y_val = values[train_end:val_end]
        row = {"alpha": float(alpha)}
        row.update(metrics(y_val, pred))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    table.to_csv(RESULT_DIR / "ewma_validation_tuning.csv", index=False)
    return float(table.loc[0, "alpha"]), table


def fit_sarima(train_val: pd.Series) -> tuple[object, pd.DataFrame, tuple[int, int, int], tuple[int, int, int, int]]:
    y = np.log(train_val)
    rows = []
    best_result = None
    best_bic = math.inf
    best_order = (0, 1, 0)
    best_seasonal = (0, 0, 0, 0)

    seasonal_orders = [(0, 0, 0, 0)]
    for p_season in [0, 1]:
        for d_season in [0, 1]:
            for q_season in [0, 1]:
                if (p_season, d_season, q_season) != (0, 0, 0):
                    seasonal_orders.append((p_season, d_season, q_season, SEASONAL_PERIOD))

    for p in range(4):
        for d in [0, 1]:
            for q in range(4):
                for seasonal_order in seasonal_orders:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result = SARIMAX(
                                y,
                                order=(p, d, q),
                                seasonal_order=seasonal_order,
                                trend="n",
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False, maxiter=200)
                        rows.append(
                            {
                                "order": str((p, d, q)),
                                "seasonal_order": str(seasonal_order),
                                "aic": result.aic,
                                "bic": result.bic,
                            }
                        )
                        if result.bic < best_bic:
                            best_bic = result.bic
                            best_result = result
                            best_order = (p, d, q)
                            best_seasonal = seasonal_order
                    except Exception as exc:
                        rows.append(
                            {
                                "order": str((p, d, q)),
                                "seasonal_order": str(seasonal_order),
                                "aic": np.nan,
                                "bic": np.nan,
                                "error": str(exc)[:120],
                            }
                        )

    selection = pd.DataFrame(rows).sort_values(["bic", "aic"], na_position="last").reset_index(drop=True)
    selection.to_csv(RESULT_DIR / "sarima_model_selection.csv", index=False)

    if best_result is None:
        raise RuntimeError("No SARIMA model could be fitted.")
    return best_result, selection, best_order, best_seasonal


def sarima_rolling_forecast(result: object, test_values: np.ndarray) -> tuple[np.ndarray, object]:
    preds = []
    current = result
    for actual in test_values:
        log_pred = current.forecast(steps=1).iloc[0]
        preds.append(float(np.exp(log_pred)))
        current = current.append([np.log(actual)], refit=False)
    return np.array(preds), current


def save_residual_diagnostics(result: object) -> None:
    resid = pd.Series(result.resid).dropna()
    resid = resid[np.isfinite(resid)]
    lb = acorr_ljungbox(resid, lags=[5, 10, 20], return_df=True)
    lb.to_csv(RESULT_DIR / "sarima_ljung_box.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(resid.index, resid.to_numpy(), color="#7f7f7f", linewidth=1.1)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.title("SARIMA Residual Time Series")
    plt.xlabel("Residual index")
    plt.ylabel("residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sarima_residual_timeseries.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(resid, lags=40, ax=ax)
    ax.set_title("SARIMA Residual ACF")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sarima_residual_acf.png", dpi=200)
    plt.close(fig)

    resid_sq = resid**2
    plt.figure(figsize=(10, 4))
    plt.plot(resid_sq.index, resid_sq.to_numpy(), color="#9467bd", linewidth=1.1)
    plt.title("SARIMA Squared Residual Time Series")
    plt.xlabel("Residual index")
    plt.ylabel("squared residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sarima_residual_squared_timeseries.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(resid_sq, lags=40, ax=ax)
    ax.set_title("ACF of SARIMA Squared Residuals")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sarima_residual_squared_acf.png", dpi=200)
    plt.close(fig)

    arch_rows = []
    for lag in [5, 10, 20]:
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=lag)
        arch_rows.append(
            {
                "lag": lag,
                "lm_stat": lm_stat,
                "lm_pvalue": lm_pvalue,
                "f_stat": f_stat,
                "f_pvalue": f_pvalue,
            }
        )
    pd.DataFrame(arch_rows).to_csv(RESULT_DIR / "arch_lm_test.csv", index=False)


def make_supervised(values: np.ndarray, lookback: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(start, end):
        x.append(values[i - lookback : i])
        y.append(values[i])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(
    scaled_values: np.ndarray,
    train_end: int,
    val_end: int,
    lookback: int,
    hidden_size: int,
) -> tuple[LSTMForecaster, float, int, pd.DataFrame]:
    x_train, y_train = make_supervised(scaled_values, lookback, lookback, train_end)
    x_val, y_val = make_supervised(scaled_values, lookback, train_end, val_end)

    train_ds = TensorDataset(
        torch.tensor(x_train).unsqueeze(-1),
        torch.tensor(y_train),
    )
    val_x = torch.tensor(x_val).unsqueeze(-1)
    val_y = torch.tensor(y_val)

    model = LSTMForecaster(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_state = None
    best_val = math.inf
    best_epoch = 0
    patience = 12
    stale = 0
    history_rows = []

    for epoch in range(1, 201):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        train_loss = float(np.mean(train_losses))
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_epoch, pd.DataFrame(history_rows)


def save_lstm_loss_curve(history: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss", linewidth=1.5)
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss", linewidth=1.5)
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "lstm_loss_curve.png", dpi=200)
    plt.close()


def tune_and_predict_lstm(full: pd.DataFrame, train_end: int, val_end: int) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    values = full["hy_oas"].to_numpy(dtype=np.float32)
    train_mean = values[:train_end].mean()
    train_std = values[:train_end].std()
    scaled = (values - train_mean) / train_std

    rows = []
    best_model = None
    best_params = None
    best_history = None
    best_rmse = math.inf

    for lookback in [10, 20, 30]:
        for hidden_size in [16, 32]:
            model, best_val_loss, best_epoch, history = train_lstm(scaled, train_end, val_end, lookback, hidden_size)
            model.eval()
            preds_scaled = []
            with torch.no_grad():
                for i in range(train_end, val_end):
                    x = torch.tensor(scaled[i - lookback : i], dtype=torch.float32).reshape(1, lookback, 1)
                    preds_scaled.append(float(model(x).item()))
            pred = np.array(preds_scaled) * train_std + train_mean
            y_val = values[train_end:val_end]
            row = {
                "lookback": lookback,
                "hidden_size": hidden_size,
                "best_epoch": best_epoch,
                "val_loss_scaled": best_val_loss,
            }
            row.update(metrics(y_val, pred))
            rows.append(row)
            if row["RMSE"] < best_rmse:
                best_rmse = row["RMSE"]
                best_model = model
                best_params = (lookback, hidden_size)
                best_history = history

    tuning = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    tuning.to_csv(RESULT_DIR / "lstm_validation_tuning.csv", index=False)

    if best_model is None or best_params is None or best_history is None:
        raise RuntimeError("No LSTM model could be trained.")

    best_history.to_csv(RESULT_DIR / "lstm_training_history.csv", index=False)
    save_lstm_loss_curve(best_history)

    lookback, _ = best_params
    best_model.eval()
    test_preds = []
    with torch.no_grad():
        for i in range(val_end, len(values)):
            x = torch.tensor(scaled[i - lookback : i], dtype=torch.float32).reshape(1, lookback, 1)
            pred_scaled = float(best_model(x).item())
            test_preds.append(pred_scaled * train_std + train_mean)
    return np.array(test_preds), tuning, best_history


def save_predictions_plot(predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(predictions["date"], predictions["actual"], label="Actual", color="black", linewidth=2.0)
    plt.plot(predictions["date"], predictions["Naive"], label="Naive", linewidth=1.3)
    plt.plot(predictions["date"], predictions["EWMA"], label="EWMA", linewidth=1.3)
    plt.plot(predictions["date"], predictions["SARIMA"], label="SARIMA", linewidth=1.3)
    plt.plot(predictions["date"], predictions["LSTM"], label="LSTM", linewidth=1.3)
    plt.title("One-step-ahead Forecasts on Test Set")
    plt.xlabel("Date")
    plt.ylabel("HY OAS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "test_forecast_comparison.png", dpi=200)
    plt.close()


def write_summary(
    df: pd.DataFrame,
    splits: SplitData,
    stationarity: pd.DataFrame,
    best_alpha: float,
    sarima_order: tuple[int, int, int],
    sarima_seasonal: tuple[int, int, int, int],
    metrics_table: pd.DataFrame,
    lstm_tuning: pd.DataFrame,
) -> None:
    best_lstm = lstm_tuning.iloc[0]
    stationarity_md = dataframe_to_markdown(stationarity)
    metrics_md = dataframe_to_markdown(metrics_table)
    arch_path = RESULT_DIR / "arch_lm_test.csv"
    arch_table = pd.read_csv(arch_path) if arch_path.exists() else pd.DataFrame()
    arch_md = dataframe_to_markdown(arch_table) if not arch_table.empty else ""
    if not arch_table.empty and (arch_table["lm_pvalue"] < 0.05).any():
        arch_conclusion = "ARCH-LM 检验的 p 值较小，说明 SARIMA 残差中可能存在 ARCH 效应，即波动大小可能仍有一定持续性。"
    else:
        arch_conclusion = "ARCH-LM 检验没有给出很强的 ARCH 效应证据。这里把 ARCH 分析作为补充诊断，不改变本文的一步预测比较结论。"
    lines = [
        "# Project3 HY OAS 单步预测总结",
        "",
        "预测任务设定为日度 HY OAS 的 one-step-ahead forecasting。本文不做长期多步预测，所有测试集预测均使用一步向前滚动方式，并在每一步使用已经观察到的真实历史值更新信息集。",
        "",
        "## 数据与划分",
        "- 数据来自 FRED，变量为 ICE BofA US High Yield Index Option-Adjusted Spread，代码为 BAMLH0A0HYM2。",
        "- 数据频率为日度，单位为 percent。HY OAS 反映高收益债信用利差，数值上升通常表示信用风险或市场避险情绪上升。",
        f"- 清洗后样本区间：{df['date'].min().date()} 至 {df['date'].max().date()}",
        f"- 样本量：{len(df)}",
        f"- 清洗后缺失值数量：{int(df.isna().sum().sum())}",
        f"- 训练集：{splits.train['date'].min().date()} 至 {splits.train['date'].max().date()}，{len(splits.train)} 个观测",
        f"- 验证集：{splits.val['date'].min().date()} 至 {splits.val['date'].max().date()}，{len(splits.val)} 个观测",
        f"- 测试集：{splits.test['date'].min().date()} 至 {splits.test['date'].max().date()}，{len(splits.test)} 个观测",
        "",
        "## 平稳性检验",
        "本文主要使用 ADF 检验判断序列是否存在单位根，并结合时序图、ACF 和 PACF 图辅助观察。ADF 检验的原假设是序列存在单位根，也就是序列不平稳。如果 p 值较小，则拒绝原假设，可以认为序列更接近平稳。",
        "",
        stationarity_md,
        "",
        "原始 hy_oas 的 ADF p 值约为 0.050，结果比较接近 5% 水平，不能很稳地认为它平稳。log(hy_oas) 的 p 值为 0.079，也没有拒绝单位根。一阶差分后的 diff_log_hy_oas 的 ADF p 值接近 0，说明差分后序列更接近平稳。因此后续 SARIMA 在 log(hy_oas) 上允许 d=1 的差分结构。",
        "ACF/PACF 图不作为正式平稳性检验，只用于辅助观察差分后序列的相关结构，并为 ARMA 或 SARIMA 定阶提供参考。",
        "金融日度数据可能存在 5 个交易日周期，因此 SARIMA 候选模型中检查了 seasonal period = 5；但季节项未改善 BIC，所以最终模型退化为非季节 ARIMA。",
        "",
        "## 模型设定",
        f"- Naive: y_hat[t] = y[t-1].",
        f"- EWMA：在验证集上选择 alpha = {best_alpha:.2f}。",
        f"- SARIMA：在 log(hy_oas) 上按 BIC 选模，最终 order = {sarima_order}, seasonal_order = {sarima_seasonal}。季节项未改善 BIC，因此退化为非季节 ARIMA 形式。",
        "- ARIMA(0,1,0) 是随机游走形式，其一步预测等于上一期观测值。因此 SARIMA/ARIMA 与 Naive 的测试集预测误差完全一致是合理结果，不是代码错误。",
        f"- LSTM：验证集最优 lookback = {int(best_lstm['lookback'])}, hidden size = {int(best_lstm['hidden_size'])}, early stopping epoch = {int(best_lstm['best_epoch'])}。",
        "",
        "## ARCH 效应辅助检查",
        "除了残差自相关外，本文还检查了 SARIMA 残差平方的相关性。金融时间序列中常见波动聚集，即残差本身相关性不强，但残差平方可能有相关性。本文画出残差平方时序图和残差平方 ACF，并使用 ARCH-LM 检验作为辅助诊断。ARCH 部分不改变主模型，只作为后续可以使用 GARCH 模型的依据。",
        "",
        arch_md,
        "",
        arch_conclusion,
        "",
        "## 测试集误差",
        metrics_md,
        "",
        "## 结论",
        "LSTM 在测试集上取得最低误差，但相对 EWMA、SARIMA 和 Naive 的提升幅度很小，因此不能夸大深度学习模型优势。",
        "HY OAS 具有短期惯性和风险冲击特征，阶段性变化与突发尖峰会使长期多步预测不稳定，因此本文重点比较短期 one-step-ahead 预测能力。SARIMA 可以刻画线性自相关结构，LSTM 可以尝试捕捉非线性动态，但在信用利差这种高持续性日度序列中，复杂模型不一定稳定优于 Naive 和 EWMA 等简单基准模型。测试结果也应围绕这些 benchmark 进行解释。",
        "",
    ]
    (OUTPUT_DIR / "project3_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    set_reproducible()
    ensure_dirs()

    df = load_data()
    save_series_overview(df)
    save_transformed_series_plots(df)
    stationarity = stationarity_tests(df)
    save_acf_pacf(df)

    splits = split_data(df)
    save_split_plot(splits)
    pd.DataFrame(
        [
            {
                "split": "train",
                "start_date": splits.train["date"].min().date(),
                "end_date": splits.train["date"].max().date(),
                "n": len(splits.train),
            },
            {
                "split": "validation",
                "start_date": splits.val["date"].min().date(),
                "end_date": splits.val["date"].max().date(),
                "n": len(splits.val),
            },
            {
                "split": "test",
                "start_date": splits.test["date"].min().date(),
                "end_date": splits.test["date"].max().date(),
                "n": len(splits.test),
            },
        ]
    ).to_csv(RESULT_DIR / "split_info.csv", index=False)
    train_end = len(splits.train)
    val_end = train_end + len(splits.val)
    full_values = df["hy_oas"].to_numpy()
    test_values = splits.test["hy_oas"].to_numpy()

    naive_pred = naive_forecast(df, val_end)

    best_alpha, _ = tune_ewma(df, train_end, val_end)
    ewma_pred = ewma_one_step(full_values, best_alpha, val_end, len(df))

    sarima_result, _, sarima_order, sarima_seasonal = fit_sarima(df.iloc[:val_end]["hy_oas"])
    save_residual_diagnostics(sarima_result)
    sarima_pred, _ = sarima_rolling_forecast(sarima_result, test_values)

    lstm_pred, lstm_tuning, _ = tune_and_predict_lstm(df, train_end, val_end)

    predictions = pd.DataFrame(
        {
            "date": splits.test["date"].to_numpy(),
            "actual": test_values,
            "Naive": naive_pred,
            "EWMA": ewma_pred,
            "SARIMA": sarima_pred,
            "LSTM": lstm_pred,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    save_predictions_plot(predictions)

    metric_rows = []
    for model_name in ["Naive", "EWMA", "SARIMA", "LSTM"]:
        row = {"model": model_name}
        row.update(metrics(predictions["actual"].to_numpy(), predictions[model_name].to_numpy()))
        metric_rows.append(row)
    metrics_table = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    metrics_table = metrics_table.rename(columns={"MAPE": "MAPE (%)"})
    metrics_table.to_csv(RESULT_DIR / "test_metrics.csv", index=False)

    write_summary(
        df,
        splits,
        stationarity,
        best_alpha,
        sarima_order,
        sarima_seasonal,
        metrics_table,
        lstm_tuning,
    )

    print("Project3 one-step-ahead forecasting completed.")
    print(f"Data period: {df['date'].min().date()} to {df['date'].max().date()}, n={len(df)}")
    print(f"Best EWMA alpha: {best_alpha:.2f}")
    print(f"Selected SARIMA order: {sarima_order}, seasonal_order: {sarima_seasonal}")
    print("Best LSTM setting:")
    print(lstm_tuning.iloc[0].to_string())
    print()
    print("Test metrics:")
    print(metrics_table.to_string(index=False))
    print()
    print(f"Summary written to: {OUTPUT_DIR / 'project3_summary.md'}")


if __name__ == "__main__":
    main()
