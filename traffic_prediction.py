"""Traffic congestion prediction: a random forest trained on synthetic data.

The data is generated from a simple rule (rush-hour peaks, weather and events
add congestion, plus noise), so the scores below show the workflow, not
real-world accuracy. Swap make_data() for a real dataset to reuse the pipeline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
FEATURES = ["hour", "weekend", "weather", "event"]  # weather: 0 clear, 1 rain, 2 fog


def congestion(hour, weekend, weather, event):
    """Underlying 'truth' used to generate the data (0-100 congestion index)."""
    peak = lambda c, s, a: a * np.exp(-((hour - c) ** 2) / (2 * s * s))
    weekday = 18 + peak(8, 1.6, 48) + peak(17.5, 2, 55) + peak(13, 2, 10)
    weekend_ = 15 + peak(13, 3.5, 30) + peak(19, 2, 12)
    base = np.where(weekend == 0, weekday, weekend_)
    return np.clip(base + np.array([0, 14, 8])[weather] + event * 18, 0, 100)


def make_data(n=1000):
    df = pd.DataFrame({
        "hour": rng.integers(0, 24, n),
        "weekend": (rng.random(n) < 0.28).astype(int),
        "weather": rng.integers(0, 3, n),
        "event": (rng.random(n) < 0.12).astype(int),
    })
    df["congestion"] = congestion(*(df[c].to_numpy() for c in FEATURES)) + rng.normal(0, 4, n)
    return df


def main():
    df = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["congestion"], test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Test R2 : {r2_score(y_test, pred):.3f}")
    print(f"Test MAE: {mean_absolute_error(y_test, pred):.2f} (index points)")
    print("\nFeature importance:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:8s} {imp:.2f}")

    scenario = pd.DataFrame([[8, 0, 1, 0]], columns=FEATURES)
    print(f"\nWeekday 08:00, rain, no event -> predicted congestion {model.predict(scenario)[0]:.0f}/100")

    hours = np.arange(24)
    for label, weather in [("Clear", 0), ("Rain", 1)]:
        grid = pd.DataFrame({"hour": hours, "weekend": 0, "weather": weather, "event": 0})
        plt.plot(hours, model.predict(grid), label=f"Weekday, {label.lower()}")
    plt.xlabel("Hour of day"); plt.ylabel("Predicted congestion (0-100)")
    plt.title("Predicted congestion by hour"); plt.legend(); plt.tight_layout()
    plt.savefig("congestion_by_hour.png", dpi=150)
    print("\nSaved chart: congestion_by_hour.png")


if __name__ == "__main__":
    main()
