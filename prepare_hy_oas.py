from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
START_DATE = "2023-06-01"
DATA_PATH = Path("data/hy_oas_2023_2026.csv")
FIGURE_PATH = Path("figures/hy_oas_timeseries.png")


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FRED_CSV_URL)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["BAMLH0A0HYM2"] = pd.to_numeric(df["BAMLH0A0HYM2"], errors="coerce")

    missing_before_drop = df.isna().sum()

    df = df.dropna()
    df = df[df["observation_date"] >= pd.Timestamp(START_DATE)]
    df = df.rename(columns={"observation_date": "date", "BAMLH0A0HYM2": "hy_oas"})
    df = df[["date", "hy_oas"]].sort_values("date").reset_index(drop=True)

    df.to_csv(DATA_PATH, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["hy_oas"], linewidth=1.5)
    plt.title("ICE BofA US High Yield Index Option-Adjusted Spread")
    plt.xlabel("Date")
    plt.ylabel("HY OAS")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()

    n = len(df)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    print("HY OAS data prepared")
    print(f"Source: {FRED_CSV_URL}")
    print(f"Saved CSV: {DATA_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")
    print()
    print("Basic information")
    print(f"Start date: {df['date'].min().date()}")
    print(f"End date: {df['date'].max().date()}")
    print(f"Sample size: {n}")
    print("Missing values after cleaning:")
    print(df.isna().sum().to_string())
    print()
    print("Missing values before drop:")
    print(missing_before_drop.to_string())
    print()
    print("Descriptive statistics:")
    print(df["hy_oas"].describe().to_string())
    print()
    print("Suggested chronological split for later univariate modeling:")
    print(f"Train: rows 0 to {train_end - 1} ({train_end} observations)")
    print(f"Validation: rows {train_end} to {val_end - 1} ({val_end - train_end} observations)")
    print(f"Test: rows {val_end} to {n - 1} ({n - val_end} observations)")
    print()
    print("Split date ranges:")
    print(f"Train: {df.loc[0, 'date'].date()} to {df.loc[train_end - 1, 'date'].date()}")
    print(f"Validation: {df.loc[train_end, 'date'].date()} to {df.loc[val_end - 1, 'date'].date()}")
    print(f"Test: {df.loc[val_end, 'date'].date()} to {df.loc[n - 1, 'date'].date()}")


if __name__ == "__main__":
    main()
