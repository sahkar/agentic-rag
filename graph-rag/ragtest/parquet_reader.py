import pandas as pd

reports = pd.read_parquet("output/community_reports.parquet")
print("REPORTS:")
print(reports.iloc[[0, 3]])

entities = pd.read_parquet("output/entities.parquet")
print("\nENTITY 19:")
print(entities.iloc[19])

relationships = pd.read_parquet("output/relationships.parquet")
print("\nRELATIONSHIP 19:")
print(relationships.iloc[19])

