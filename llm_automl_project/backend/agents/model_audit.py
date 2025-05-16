def audit_model_accuracy(history):
    issues = []
    if history['accuracy'] < 0.75:
        issues.append("⚠️ Accuracy dropped below threshold.")
    return issues

def audit_bias_metrics(metrics):
    bias_alerts = []
    if abs(metrics.get("Statistical Parity", 0)) > 0.2:
        bias_alerts.append("⚠️ Potential bias detected.")
    return bias_alerts
