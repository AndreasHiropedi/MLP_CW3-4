import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../datasets/implicit_hate_v1_stg0-2_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_hate = LabelEncoder()
data['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data['hate_or_not_hate'])

# Only consider rows with hate for the second label to avoid NaN issues
data_hate_only = data[data['hate_or_not_hate'] == 'hate'].dropna(subset=['implicit_or_explicit'])
label_encoder_imp_exp = LabelEncoder()
data_hate_only['implicit_or_explicit_encoded'] = label_encoder_imp_exp.fit_transform(data_hate_only['implicit_or_explicit'])

# Step 2: Split the data for the first task
X = data['post']  # Features for the first model
y = data['hate_or_not_hate_encoded']  # Labels for the first model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer_full = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = tfidf_vectorizer_full.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer_full.transform(X_test)

# Train the first SVC model
svc_model_1 = SVC()
svc_model_1.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred_1 = svc_model_1.predict(X_test_tfidf)

# Evaluate the first model
_accuracy_1 = accuracy_score(y_test, y_pred_1)
_classification_rep_1 = classification_report(y_test, y_pred_1, target_names=label_encoder_hate.classes_)

# Filter out 'hate' posts for the second task using the entire dataset
# This is to ensure we have a clean separation for training the second model
X_hate_only = tfidf_vectorizer_full.transform(data_hate_only['post'])
y_hate_only = data_hate_only['implicit_or_explicit_encoded']

# Split the 'hate' data into training and testing sets for the second model
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_hate_only, y_hate_only, test_size=0.2, random_state=42)

# Train the second SVC model
svc_model_2 = SVC()
svc_model_2.fit(X_train_2, y_train_2)

# Predict on the test set for the second model
y_pred_2 = svc_model_2.predict(X_test_2)

# Evaluate the second model
_accuracy_2 = accuracy_score(y_test_2, y_pred_2)
_classification_rep_2 = classification_report(y_test_2, y_pred_2, target_names=label_encoder_imp_exp.classes_)

# Identify 'implicit_hate' predictions from the second model
# Assuming 'data_hate_only' has a column 'implicit_class' for these entries
# We'll simulate this filtering based on the actual labels first, to prepare for training
implicit_hate_predictions = y_pred_2 == label_encoder_imp_exp.transform(['implicit_hate'])[0]
X_implicit_hate = X_test_2[implicit_hate_predictions]

# For training purposes, we use the actual labels to create a filtered dataset of implicit hate
data_implicit_hate_only = data_hate_only[data_hate_only['implicit_or_explicit'] == 'implicit_hate']

# Prepare labels for 'implicit_class'
label_encoder_implicit_class = LabelEncoder()
data_implicit_hate_only['implicit_class_encoded'] = label_encoder_implicit_class.fit_transform(data_implicit_hate_only['implicit_class'])

# Split the 'implicit_hate' data into training and testing sets for the third model
X_implicit = tfidf_vectorizer_full.transform(data_implicit_hate_only['post'])
y_implicit = data_implicit_hate_only['implicit_class_encoded']

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_implicit, y_implicit, test_size=0.2, random_state=42)

# Train the third SVC model
svc_model_3 = SVC()
svc_model_3.fit(X_train_3, y_train_3)

# Predict on the test set for the third model
y_pred_3 = svc_model_3.predict(X_test_3)

# Evaluate the third model
accuracy_3 = accuracy_score(y_test_3, y_pred_3)
classification_rep_3 = classification_report(y_test_3, y_pred_3, target_names=label_encoder_implicit_class.classes_)

print(accuracy_3, classification_rep_3)
