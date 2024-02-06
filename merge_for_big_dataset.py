import pandas as pd

df_stg0 = pd.read_csv("implicit_hate_v1_stg0_posts.tsv", delimiter='\t')

df_stg1 = pd.read_csv("implicit_hate_v1_stg1_posts.tsv", delimiter='\t')

df_stg2 = pd.read_csv("implicit_hate_v1_stg2_posts.tsv", delimiter='\t')

# Add implicit versus explicit to hate/ no hate data
merged_stg0_stg1 = pd.merge(df_stg0, df_stg1, on='post', how='left')

merged_stg0_stg1.rename(columns={'class_x': 'hate_or_not_hate', 'class_y': 'implicit_or_explicit'}, inplace=True)

new_column = merged_stg0_stg1['implicit_or_explicit'].apply(lambda x: "" if x == 'not_hate' else x)

merged_stg0_stg1['implicit_or_explicit'] = new_column

# Add implicit types to previously merged data
merged_with_stg2 = pd.merge(merged_stg0_stg1, df_stg2, on='post', how='left')

# Count rows with missing implicit_class despite implicit hate
condition = merged_with_stg2['implicit_class'].isna() & merged_with_stg2['extra_implicit_class'].isna() \
            & (merged_with_stg2['implicit_or_explicit'] == 'implicit_hate')

# Keep rows that meet the condition
merged_with_stg2_filtered = merged_with_stg2[~condition]

merged_with_stg2_filtered.to_csv('implicit_hate_v1_stg0-2_posts.tsv', sep='\t', index=False, header=True)
