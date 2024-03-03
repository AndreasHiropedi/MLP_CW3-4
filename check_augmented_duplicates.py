import pandas as pd

# Load the dataset
data = pd.read_csv('final_augmented_dataset.tsv', delimiter='\t')

irony_data = data[data['implicit_class'] == 'irony']
other_data = data[data['implicit_class'] == 'other']
inferiority_data = data[data['implicit_class'] == 'inferiority']
threatening_data = data[data['implicit_class'] == 'threatening']
explicit_data = data[data['implicit_or_explicit'] == 'explicit_hate']

unique_irony_posts = len(irony_data['post'].dropna().unique())
unique_inferiority_posts = len(inferiority_data['post'].dropna().unique())
unique_other_posts = len(other_data['post'].dropna().unique())
unique_explicit_posts = len(explicit_data['post'].dropna().unique())
unique_threatening_posts = len(threatening_data['post'].dropna().unique())

print('Irony unique: ', unique_irony_posts)
print('Other unique: ', unique_other_posts)
print('Threatening unique: ', unique_threatening_posts)
print('Inferiority unique: ', unique_inferiority_posts)
print('Explicit unique: ', unique_explicit_posts)


"""
Results show the following:

    - other count for unique values is still low: 157, need to get it closer to 1100 maybe (so closer look)
    - explicit counts for unique values is also quite low: 2173, need to get it closer to 5000 maybe (so closer look)
    - remaining classes were augmented well, so no need to worry about them
"""
