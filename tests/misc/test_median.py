#!/usr/bin/env python3
import statistics

# Test data
ages = [25, 30, 22, 28]
scores = [95, 85, 90, 88]

# Calculate medians
age_median = statistics.median(ages)
score_median = statistics.median(scores)

# Print results
print(f"Age median: {age_median}")
print(f"Score median: {score_median}")
