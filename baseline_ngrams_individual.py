import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

######################################### Classifier 1 #########################################

data_c1 = pd.read_csv("implicit_hate_v1_stg0_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_hate = LabelEncoder()
data_c1['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data_c1['class'])

# Split the data
X_1 = data_c1['post']  # Features for the first model
y_1 = data_c1['hate_or_not_hate_encoded']  # Labels for the first model

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)

# Create a pipeline
pipeline_1 = make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),  # Using unigrams
    SVC()
)

# Train the model
pipeline_1.fit(X_train_1, y_train_1)

# Predict
predictions_1 = pipeline_1.predict(X_test_1)

# Reverse encoding for interpretation
predicted_labels_1 = label_encoder_hate.inverse_transform(predictions_1)
true_labels_1 = label_encoder_hate.inverse_transform(y_test_1)

# Calculate accuracy
accuracy_1 = accuracy_score(y_test_1, predictions_1)

# Generate classification report
classification_rep_1 = classification_report(true_labels_1, predicted_labels_1)

print('Hate versus not hate classifier results')
print()
print(accuracy_1)
print()
print(classification_rep_1)
print()
print('----------------------------------------')
print()

######################################### Classifier 2 #########################################

data_c2 = pd.read_csv("implicit_hate_v1_stg1_posts.tsv", delimiter='\t')

# Remove rows with label 'not_hate'
data_c2 = data_c2[data_c2['class'] != 'not_hate'].dropna()

# Encode the labels
label_encoder_implicit = LabelEncoder()
data_c2['explicit_or_implicit_encoded'] = label_encoder_implicit.fit_transform(data_c2['class'])

# Split the data
X_2 = data_c2['post']  # Features for the first model
y_2 = data_c2['explicit_or_implicit_encoded']  # Labels for the first model

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

# Create a pipeline
pipeline_2 = make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),  # Using unigrams
    SVC()
)

# Train the model
pipeline_2.fit(X_train_2, y_train_2)

# Predict
predictions_2 = pipeline_2.predict(X_test_2)

# Reverse encoding for interpretation
predicted_labels_2 = label_encoder_implicit.inverse_transform(predictions_2)
true_labels_2 = label_encoder_implicit.inverse_transform(y_test_2)

# Calculate accuracy
accuracy_2 = accuracy_score(y_test_2, predictions_2)

# Generate classification report
classification_rep_2 = classification_report(true_labels_2, predicted_labels_2)

print('Implicit versus explicit classifier results')
print()
print(accuracy_2)
print()
print(classification_rep_2)
print()
print('----------------------------------------')
print()

######################################### Classifier 3 #########################################

data_c3 = pd.read_csv("implicit_hate_v1_stg2_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_implicit_types = LabelEncoder()
data_c3['implicit_types_encoded'] = label_encoder_implicit_types.fit_transform(data_c3['implicit_class'])

# Split the data for the first task
X_3 = data_c3['post']  # Features for the first model
y_3 = data_c3['implicit_types_encoded']  # Labels for the first model

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=42)

# Create a pipeline
pipeline_3 = make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),  # Using unigrams
    SVC()
)

# Train the model
pipeline_3.fit(X_train_3, y_train_3)

# Predict
predictions_3 = pipeline_3.predict(X_test_3)

# Reverse encoding for interpretation
predicted_labels_3 = label_encoder_implicit_types.inverse_transform(predictions_3)
true_labels_3 = label_encoder_implicit_types.inverse_transform(y_test_3)

# Calculate accuracy
accuracy_3 = accuracy_score(y_test_3, predictions_3)

# Generate classification report
classification_rep_3 = classification_report(true_labels_3, predicted_labels_3)

print('Implicit classes classifier results')
print()
print(accuracy_3)
print()
print(classification_rep_3)
print()
print('----------------------------------------')
print()
