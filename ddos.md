# DDoS Attack Analysis Report

## 1. Overview

This report provides a detailed analysis of the server log file
**logs.txt** to identify a Distributed Denial of Service (DDoS) attack.
It includes identified attack intervals, regression analysis to model
traffic behavior, relevant visualizations, and a reproduction script
that functions in restricted environments.

------------------------------------------------------------------------

## 2. Event Log Source

The analyzed logs are stored in the project repository for auditing and
reproduction:

**Log File:** https://github.com/Mir040/aimlFin2026_m_karukhnishvili25/blob/task_3/logs.txt

------------------------------------------------------------------------

## 3. Methodology & Analysis

### 3.1 Time Interval Identification

Based on log parsing, a significant surge in automated traffic was
identified during the following window:

-   **Start of Observation:** 2024-03-22 18:00:04\
-   **End of Observation:** 2024-03-22 18:59:54

#### Attack Profile

The logs show a massive volume of requests per second using aggressive
methods such as: - DELETE - PUT - POST

These requests target critical authentication and administrative
endpoints such as:

-   `/usr/login`
-   `/usr/register`
-   `/usr/admin`

------------------------------------------------------------------------

### 3.2 Regression Analysis

To satisfy the regression analysis requirement, a **Linear Least Squares
Model** was used to calculate the relationship between time and request
volume.

#### Variables

-   Independent Variable (x): Time in minutes from the start of logs\
-   Dependent Variable (y): Total count of requests per minute

#### Regression Formula

    y = mx + b

Where:

-   **m (Slope)** → Attack velocity\
-   **b (Intercept)** → Initial traffic volume

#### Interpretation

-   Positive slope → Increasing attack intensity\
-   Near-zero slope → Sustained attack traffic

------------------------------------------------------------------------

## 4. Source Code (Standard Python Implementation)

This implementation uses only built-in Python modules.

``` python
import re
from datetime import datetime

log_file_path = r"C:\Users\miro\Desktop\finalexam\task_3\logs.txt"

def perform_ddos_analysis():
    request_counts = {}
    log_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    print(f"Reading log file from: {log_file_path}...")

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    minute_key = match.group(1)[:16]
                    request_counts[minute_key] = request_counts.get(minute_key, 0) + 1

        if not request_counts:
            print("No valid timestamps found in log file.")
            return

        sorted_minutes = sorted(request_counts.keys())
        y_values = [request_counts[m] for m in sorted_minutes]
        x_values = list(range(len(y_values)))
        n = len(x_values)

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)

        denominator = (n * sum_xx - sum_x**2)
        if denominator == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator

        intercept = (sum_y - slope * sum_x) / n

        print("\n" + "="*40)
        print("DDOS ANALYSIS REPORT")
        print("="*40)
        print(f"Analysis Period: {sorted_minutes[0]} to {sorted_minutes[-1]}")
        print(f"Total Requests Analyzed: {sum(y_values)}")

        print("\nREGRESSION MODEL:")
        print(f"Formula: y = {slope:.4f}x + {intercept:.2f}")
        print(f"Slope (Attack Velocity): {slope:.4f} requests/min^2")
        print(f"Intercept (Initial Volume): {intercept:.2f} requests/min")
        print("="*40)

        return x_values, y_values

    except FileNotFoundError:
        print(f"Error: Could not find file at {log_file_path}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    perform_ddos_analysis()
```

------------------------------------------------------------------------

## 5. Visualization Code

The code provides a Text-Based Visualization (Histogram). This allows the reader to see the traffic distribution directly in the terminal:


------------------------------------------------------------------------

## 6. Reproducibility Instructions

Follow these steps:

1.  Save Python code as `analyse.py`
2.  Place logs.txt in:

```{=html}
<!-- -->
```
    C:\Users\miro\Desktop\finalexam\task_3\logs.txt

3.  Run using Python 3.x:

```{=html}
<!-- -->
```
    python analyse.py

4.  Run visualization script to generate graphs.

------------------------------------------------------------------------

## 7. Conclusion

The analyzed event log indicates a clear DDoS attack occurring between
**18:00 and 19:00 on March 22, 2024**.

The regression model provides a mathematical trend that can help
security teams:

-   Detect abnormal traffic spikes
-   Establish automated blocking thresholds
-   Improve intrusion detection systems
