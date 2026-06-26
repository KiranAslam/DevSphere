import pandas as pd

df = pd.read_csv("hotel_bookings - Copy.csv")

print("==First 5 Rows of dataset==")
df.head(5)

print(f"==Dataset Info==\n{df.info()}")

print(f"==Statistics Summary==\n")
print(df.describe(include='all'))

print("==Missing Values==")
print(df.isnull().sum())

print("==Duplicate Values==")
print(df.duplicated())

#Data Cleaning

df.drop(columns=["company"])
print("droping company column due to excessive missing values.")

df['agent'].fillna(0)

df["country"].fillna('unknown')

df["children"].fillna(df["children"].mode()[0])


output_file = 'hotel_bookings_cleaned.csv'
df.to_csv(output_file, index=False)
print(f"\n Cleaned dataset saved as '{output_file}'")