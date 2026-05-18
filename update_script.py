import sys

new_section = """# ============================================================
# 6. REVENUE BIN CALIBRATION PLOTS
# ============================================================

for plot_type in ["linear", "linear_zoomed", "log", "log_zoomed"]:
    plt.figure(figsize=(9, 6))
    actual_plotted = False

    for model_name, df in model_data.items():
        df_cal = df[["y_true", "pred"]].copy()

        df_cal["bin"] = pd.qcut(
            df_cal["pred"].rank(method="first"),
            q=10,
            labels=False
        )

        df_cal["log_y_true"] = np.log1p(df_cal["y_true"])
        df_cal["log_pred"] = np.log1p(df_cal["pred"])

        grouped = df_cal.groupby("bin").agg(
            avg_actual=("y_true", "mean"),
            avg_predicted=("pred", "mean"),
            avg_log_actual=("log_y_true", "mean"),
            avg_log_predicted=("log_pred", "mean")
        ).reset_index()

        x = grouped["bin"] + 1

        if plot_type.startswith("linear"):
            y_act = grouped["avg_actual"]
            y_pred = grouped["avg_predicted"]
            y_label = "Average revenue"
            title = "Revenue calibration by prediction decile (Linear)"
        else:
            y_act = grouped["avg_log_actual"]
            y_pred = grouped["avg_log_predicted"]
            y_label = "Average revenue, log(1 + y)"
            title = "Revenue calibration by prediction decile (Log Scale)"

        if not actual_plotted:
            plt.plot(x, y_act, label="Actual average revenue", linewidth=2, color="black", linestyle="--")
            actual_plotted = True

        plt.plot(x, y_pred, label=f"Predicted: {model_name}", linewidth=1)

    plt.xlabel("Prediction decile, low to high")
    plt.ylabel(y_label)

    if "zoomed" in plot_type:
        plt.xlim(6, 10)
        title += " - Zoomed"
        if plot_type == "linear_zoomed":
            plt.ylim(bottom=np.exp(6), top=np.exp(10))
        elif plot_type == "log_zoomed":
            plt.ylim(bottom=6, top=10)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"calibration_deciles_all_models_{plot_type}.png", dpi=300)
    plt.close()


"""

with open("/Users/vidarelfving/data_analytics/AdvancedAnalytics/task1/models/model_visualizations.py", "r") as f:
    lines = f.readlines()

new_lines = lines[:285] + [new_section] + lines[395:]

with open("/Users/vidarelfving/data_analytics/AdvancedAnalytics/task1/models/model_visualizations.py", "w") as f:
    f.writelines(new_lines)
