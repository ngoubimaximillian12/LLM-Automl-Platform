import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_fairness_metrics(metrics: dict, output_dir="data/eda_report"):
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []

    # 📊 Standard Bar Chart
    plt.figure(figsize=(8, 4))
    keys, values = list(metrics.keys()), list(metrics.values())
    bars = plt.bar(keys, values, color="skyblue")
    plt.title("Bias & Fairness Metrics")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.05, yval, f"{yval:.2f}")
    bar_path = os.path.join(output_dir, "fairness_bar_chart.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    chart_paths.append(bar_path)
    plt.close()

    # 🔵 Optional: Pie Chart for label balance if proportions exist
    if "class_0_ratio" in metrics and "class_1_ratio" in metrics:
        plt.figure()
        sizes = [metrics["class_0_ratio"], metrics["class_1_ratio"]]
        labels = ['Class 0', 'Class 1']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["lightcoral", "lightgreen"])
        plt.axis('equal')
        pie_path = os.path.join(output_dir, "label_distribution_pie.png")
        plt.savefig(pie_path)
        chart_paths.append(pie_path)
        plt.close()

    # 📉 Optional: Line Chart for fairness evolution (if provided)
    if "fairness_trend" in metrics and isinstance(metrics["fairness_trend"], list):
        trend = metrics["fairness_trend"]
        plt.figure()
        plt.plot(range(len(trend)), trend, marker="o")
        plt.title("Fairness Over Time")
        plt.xlabel("Retrain Iteration")
        plt.ylabel("Fairness Score")
        line_path = os.path.join(output_dir, "fairness_trend.png")
        plt.savefig(line_path)
        chart_paths.append(line_path)
        plt.close()

    return chart_paths
