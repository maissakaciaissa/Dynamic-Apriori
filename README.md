# Dynamic Apriori: Adaptive Minimum Support for Association Rule Mining

A Python implementation comparing traditional fixed minimum support Apriori algorithm with a novel dynamic weighted percentile approach for mining association rules from medical data.

## Overview

This project implements and compares two approaches to frequent itemset mining:

- **Classic Apriori** with fixed minimum support threshold
- **Dynamic Weighted Apriori** with adaptive minimum support based on data distribution

The dynamic approach adjusts the minimum support threshold using weighted percentiles (Q1 and Q2) to better capture both common and rare patterns in the data.

## Dataset

The implementation uses the Heart Disease dataset (`heart.csv`), analyzing patterns in cardiovascular health indicators including:

- Age, Resting Blood Pressure, Cholesterol
- Maximum Heart Rate, ST Depression (Oldpeak)
- Various categorical health metrics

## Features

- **Data Preprocessing**: Handles missing values and discretizes numerical features into categorical bins
- **Transaction Encoding**: Transforms tabular data into transaction format suitable for association rule mining
- **Dynamic Minimum Support**: Calculates adaptive thresholds using weighted percentiles (Q1 + Q2)/2
- **Performance Metrics**: Measures execution time and memory usage for both algorithms
- **Association Rules**: Extracts rules with lift and confidence metrics
- **Visualization**: Generates comparative plots for itemsets, lift, and confidence

## Requirements

```
pandas
numpy
mlxtend
psutil
matplotlib
```

## Installation

```bash
pip install pandas numpy mlxtend psutil matplotlib
```

## Usage

```bash
python tp1.py
```

The script will:

1. Load and preprocess the heart disease dataset
2. Run both fixed and dynamic Apriori algorithms
3. Generate association rules
4. Output CSV files with frequent itemsets
5. Display comparative visualizations

## Output Files

- `frequent_itemsets_fixed.csv`: Itemsets found with fixed minsup=0.3
- `frequent_itemsets_dynamic_weighted.csv`: Itemsets found with dynamic approach

## Key Results

The dynamic weighted approach demonstrates:

- **Higher confidence** in extracted rules (better reliability)
- **Fewer but more meaningful** patterns (reduced noise)
- **Adaptive behavior** across different data distributions
- **Balance between rare and common patterns**

## Algorithm Comparison

| Metric             | Fixed Minsup (0.3) | Dynamic Weighted |
| ------------------ | ------------------ | ---------------- |
| Number of Itemsets | Higher             | Lower            |
| Average Lift       | Higher             | Moderate         |
| Average Confidence | Lower              | Higher           |
| Adaptability       | Static             | Dynamic          |

## Why Dynamic Approach?

1. **Distribution-Aware**: Adjusts to actual data frequencies
2. **Robust to Outliers**: Weighted percentiles reduce impact of extreme values
3. **Captures Rare Patterns**: Lower thresholds preserve interesting rare associations
4. **Scalable**: Adapts to different dataset sizes
5. **Clamped Range**: Maintains reasonable bounds (0.01 to 0.5)

## Visualizations

The project generates three main visualizations:

- Comparison of frequent itemset counts
- Average lift of association rules
- Average confidence of association rules
- Impact of static minsup thresholds on rule generation
