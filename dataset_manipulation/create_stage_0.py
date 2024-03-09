import pandas as pd

df = pd.read_csv("../datasets/implicit_hate_v1_stg1_posts.tsv", delimiter='\t')

new_class_column = df['class'].apply(lambda x: "hate" if x == 'implicit_hate' or x == 'explicit_hate' else 'not_hate')

df['class'] = new_class_column

df.to_csv('implicit_hate_v1_stg0_posts.tsv', sep='\t', index=False, header=True)
