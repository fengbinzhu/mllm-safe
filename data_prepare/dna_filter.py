import pandas as pd

data_file = "/data/huangyoucheng/mm-safety/data_prepare/do-not-answer/data_en.csv"
df = pd.read_csv(data_file)
harmful_types = set(df['types_of_harm'])
print(harmful_types)
