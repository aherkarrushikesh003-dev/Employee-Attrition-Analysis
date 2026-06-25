import pandas as pd

df = pd.read_csv(r"C:\Users\RYZEN 3\Employee-Attrition-Analysis\data\WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
import pandas as pd

df = pd.read_csv(r"C:\Users\RYZEN 3\Employee-Attrition-Analysis\data\WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Total Missing Values:")
print(df.isnull().sum().sum())