import re
from datetime import datetime

# UPDATED: Path to the log file as requested
log_file_path = r"C:\Users\miro\Desktop\finalexam\task_3\logs.txt"


def perform_ddos_analysis():
    request_counts = {}
    # Regex to extract the timestamp [YYYY-MM-DD HH:MM:SS]
    log_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    print(f"Reading log file from: {log_file_path}...")

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    # Use YYYY-MM-DD HH:MM as a key to group by minute
                    minute_key = match.group(1)[:16]
                    request_counts[minute_key] = request_counts.get(minute_key, 0) + 1

        if not request_counts:
            print("No valid timestamps found in log file.")
            return

        # Prepare Data for Regression
        sorted_minutes = sorted(request_counts.keys())
        y_values = [request_counts[m] for m in sorted_minutes]
        x_values = list(range(len(y_values)))  # Minutes 0, 1, 2...
        n = len(x_values)

        # Manual Linear Regression Calculation (y = mx + b)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)

        # Calculate Slope (m) and Intercept (b)
        denominator = (n * sum_xx - sum_x ** 2)
        if denominator == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator

        intercept = (sum_y - slope * sum_x) / n

        # Output Results
        print("\n" + "=" * 40)
        print("DDOS ANALYSIS REPORT")
        print("=" * 40)
        print(f"Analysis Period: {sorted_minutes[0]} to {sorted_minutes[-1]}")
        print(f"Total Requests Analyzed: {sum(y_values)}")
        print(f"\nREGRESSION MODEL:")
        print(f"Formula: y = {slope:.4f}x + {intercept:.2f}")
        print(f"Slope (Attack Velocity): {slope:.4f} requests/min^2")
        print(f"Intercept (Initial Volume): {intercept:.2f} requests/min")
        print("=" * 40)

    except FileNotFoundError:
        print(f"Error: Could not find file at {log_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    perform_ddos_analysis()