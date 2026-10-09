import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def create_chart_image(df: pd.DataFrame, asset_name: str, tf_label="5m") -> io.BytesIO:
    plot_df = df.tail(100).copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    if 'local_support' in df.columns and 'local_resistance' in df.columns:
        last_sup = df['local_support'].iloc[-1]
        last_res = df['local_resistance'].iloc[-1]
    else:
        last_sup = df['low'].rolling(window=40).min().iloc[-1]
        last_res = df['high'].rolling(window=40).max().iloc[-1]

    # Малювання свічок
    for i in range(len(plot_df)):
        op = plot_df['open'].iloc[i]
        hi = plot_df['high'].iloc[i]
        lo = plot_df['low'].iloc[i]
        cl = plot_df['close'].iloc[i]

        color = "#26a69a" if cl >= op else "#ef5350"
        ax.vlines(i, lo, hi, color=color, linewidth=1, alpha=0.9)
        ax.bar(i, abs(cl - op), bottom=min(op, cl), color=color, width=0.6, alpha=0.9)

    # Ковзні та індикатори
    if 'ema_20' in plot_df.columns:
        ax.plot(plot_df.index, plot_df['ema_20'], color="#2962FF", linestyle="-", linewidth=1.5, alpha=0.8, label="EMA 20")

    if 'bb_upper' in plot_df.columns and 'bb_lower' in plot_df.columns:
        ax.plot(plot_df.index, plot_df['bb_upper'], color="#ab47bc", linestyle="--", linewidth=1, alpha=0.6, label="BB Upper")
        ax.plot(plot_df.index, plot_df['bb_lower'], color="#ab47bc", linestyle="--", linewidth=1, alpha=0.6, label="BB Lower")

    # Рівні підтримки та опору
    ax.axhline(y=last_sup, color="#00897b", linestyle="--", alpha=0.8, linewidth=1.2, label="Support")
    ax.axhline(y=last_res, color="#c62828", linestyle="--", alpha=0.8, linewidth=1.2, label="Resistance")

    # Маркер останньої ціни входу
    last_idx = len(plot_df) - 1
    last_close = plot_df['close'].iloc[-1]
    ax.scatter(last_idx, last_close, color="#FFD700", s=60, zorder=5, label="Entry Point")

    ax.set_title(f"📊 {asset_name} [{tf_label}]", fontsize=11, color="white", weight="bold")
    ax.grid(True, color="#2a2e39", alpha=0.4)
    ax.set_facecolor("#131722")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2e39")

    fig.patch.set_facecolor("#131722")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf
