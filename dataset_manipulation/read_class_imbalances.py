import pandas as pd

# Load the dataset
data = pd.read_csv('../datasets/implicit_hate_v1_stg0-2_posts.tsv', delimiter='\t')

# Calculate counts for unique values excluding NaN for each of the remaining columns
unique_value_counts = {column: data[column].dropna().value_counts() for column in data.columns if column != 'post' and column != 'extra_implicit_class'}

print(unique_value_counts)

"""
Outcomes of running this code:

    - balance inferiority, irony, threatening and other classes so they have similar counts (around 1000 for each after balancing)
    - after that, note the counts for explicit versus implicit, and balance those by increasing explicit count accordingly
    - check hate versus no hate counts, and maybe balance those as well 
    
So, after doing the maths:

    - bump up inferiority, irony, threatening and other classes to 1100 points each
    - bump up explicit class to 5000 points
    - that leaves us with quite balanced data for hate versus no hate
"""
