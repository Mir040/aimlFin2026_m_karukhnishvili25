
# DDoS Attack Analysis Report

## Log File
task_3/logs.txt

## Detection Method
Threshold = Mean + 3 * STD

Threshold Value:
55.43


### Detected Attack Interval
Start: 2024-03-22 18:16:07
End: 2024-03-22 18:20:59

## Visualizations

### Traffic
![Traffic](traffic_plot.png)

### Regression
![Regression](regression_plot.png)

## Main Source Code Fragments

```python
threshold = traffic["requests"].mean() + 3 * traffic["requests"].std()

model = LinearRegression()
model.fit(X, y)
traffic["regression"] = model.predict(X)
```

## Reproducibility
1. Place logs.txt in task_3 folder
2. Run Python script
3. Check generated plots and intervals
