import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

######################################### Classifier 1 #########################################

data_c1 = pd.read_csv("implicit_hate_v1_stg0_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_hate = LabelEncoder()
data_c1['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data_c1['class'])

# Split the data
X = data_c1['post']  # Features for the first model
y = data_c1['hate_or_not_hate_encoded']  # Labels for the first model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer_full = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = tfidf_vectorizer_full.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer_full.transform(X_test)

# Train the SVC model
svc_model_1 = SVC()
svc_model_1.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred_1 = svc_model_1.predict(X_test_tfidf)

# Evaluate the model
accuracy_1 = accuracy_score(y_test, y_pred_1)
classification_rep_1 = classification_report(y_test, y_pred_1, target_names=label_encoder_hate.classes_)

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

# Vectorize the text data
tfidf_vectorizer_full_2 = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf_2 = tfidf_vectorizer_full_2.fit_transform(X_train_2)
X_test_tfidf_2 = tfidf_vectorizer_full_2.transform(X_test_2)

# Train the SVC model
svc_model_2 = SVC()
svc_model_2.fit(X_train_tfidf_2, y_train_2)

# Predict on the test set
y_pred_2 = svc_model_2.predict(X_test_tfidf_2)

# Evaluate the model
accuracy_2 = accuracy_score(y_test_2, y_pred_2)
classification_rep_2 = classification_report(y_test_2, y_pred_2, target_names=label_encoder_implicit.classes_)

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

# Vectorize the text data
tfidf_vectorizer_full_3 = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf_3 = tfidf_vectorizer_full_3.fit_transform(X_train_3)
X_test_tfidf_3 = tfidf_vectorizer_full_3.transform(X_test_3)

# Train the SVC model
svc_model_3 = SVC()
svc_model_3.fit(X_train_tfidf_3, y_train_3)

# Predict on the test set
y_pred_3 = svc_model_3.predict(X_test_tfidf_3)

# Evaluate the model
accuracy_3 = accuracy_score(y_test_3, y_pred_3)
classification_rep_3 = classification_report(y_test_3, y_pred_3, target_names=label_encoder_implicit_types.classes_)

print('Implicit classes classifier results')
print()
print(accuracy_3)
print()
print(classification_rep_3)
print()
print('----------------------------------------')
print()
