import pandas as pd
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load the datasets
full_data = pd.read_csv('final_augmented_dataset.tsv', delimiter='\t')
normal_data = pd.read_csv('implicit_hate_v1_stg0-2_posts.tsv', delimiter='\t')

# Split off augmented data

# Performing a merge to find only augmented rows
result = full_data.merge(normal_data, on=['post', 'hate_or_not_hate', 'implicit_or_explicit', 'implicit_class'], how='left', indicator=True).loc[lambda x: x['_merge'] == 'left_only']

# Dropping the indicator column as it's no longer needed
augmented_data = result.drop(columns=['_merge'])

# Create UMAP plots for all different augmented classes

# Initializing the TF-IDF Vectorizer and UMAP model
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)

# Re-filtering the dataset for the 'explicit hate' class only, skipping the header row and selecting the 'post' column
normal_explicit_data = normal_data[normal_data['implicit_or_explicit'] == 'explicit_hate'].iloc[1:, 0].values
augmented_explicit_data = augmented_data[augmented_data['implicit_or_explicit'] == 'explicit_hate'].iloc[1:, 0].values

# Vectorizing the filtered text data using TF-IDF
explicit_text_vectors = tfidf_vectorizer.fit_transform(normal_explicit_data)
augmented_explicit_text_vectors = tfidf_vectorizer.fit_transform(augmented_explicit_data)

# Applying UMAP on the filtered and vectorized text data
explicit_umap_embeddings = umap_model.fit_transform(explicit_text_vectors)
augmented_explicit_umap_embeddings = umap_model.fit_transform(augmented_explicit_text_vectors)

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(explicit_umap_embeddings[:, 0], explicit_umap_embeddings[:, 1], color='red', alpha=0.5, label='Standard Data')
plt.scatter(augmented_explicit_umap_embeddings[:, 0], augmented_explicit_umap_embeddings[:, 1], color='blue', alpha=0.5, label='Augmented Data')
plt.title('UMAP Visualization of Explicit Hate Speech Data')
plt.legend()
plt.show()
