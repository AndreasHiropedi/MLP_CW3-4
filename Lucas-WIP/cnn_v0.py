import pandas as pd
import numpy as np
import math
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score, precision_score, recall_score

# Load dataset
df = pd.read_csv('datasets/final_augmented_dataset.tsv', sep='\t')

# Encoding categorical variable
label_encoder_hate = LabelEncoder()
df['hate_or_not_hate'] = label_encoder_hate.fit_transform(df['hate_or_not_hate'])

# Tokenizing text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['post'])
sequences = tokenizer.texts_to_sequences(df['post'])
data = pad_sequences(sequences, maxlen=100)

# Model 1: Hate or Not Hate
X_train, X_test, y_train, y_test = train_test_split(data, df['hate_or_not_hate'], test_size=0.2, random_state=42)
model_1 = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=1, activation='sigmoid')
])
# Modify the model compilation to include Precision and Recall metrics
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Start timing your model training
start_time = time.time()
model_1.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
training_time = time.time() - start_time

# Evaluate the model to get loss and accuracy directly
eval_results = model_1.evaluate(X_test, y_test)
eval_loss, eval_accuracy, eval_precision, eval_recall = eval_results

# Use the model to make predictions on the test set
predictions_1 = model_1.predict(X_test) > 0.5

# Calculate F1 score using predictions
eval_f1 = f1_score(y_test, predictions_1, average='binary')

# Construct the result dictionary
evaluation_results_1 = {
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,
    'eval_f1': eval_f1,
    'eval_precision': eval_precision,
    'eval_recall': eval_recall,
    'eval_runtime': training_time,
    'eval_samples_per_second': len(X_test) / training_time,
    'epoch': 20 
}

# Model 2: Implicit or Explicit
df = pd.read_csv('datasets/final_augmented_dataset.tsv', sep='\t')

df['implicit_or_explicit'] = df['implicit_or_explicit'].fillna('none')
label_encoder_imp_exp = LabelEncoder()
df['implicit_or_explicit'] = label_encoder_imp_exp.fit_transform(df['implicit_or_explicit'])

df_hate = df[df['hate_or_not_hate'] == 'hate']

tokenizer.fit_on_texts(df_hate['post'])
sequences = tokenizer.texts_to_sequences(df_hate['post'])
data = pad_sequences(sequences, maxlen=100)
X_train, X_test, y_train, y_test = train_test_split(data, df_hate['implicit_or_explicit'], test_size=0.2, random_state=42)
model_2 = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=1, activation='sigmoid')
])

# Modify the model compilation to include Precision and Recall metrics for model_2
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Start timing your model training for model_2
start_time = time.time()
model_2.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
training_time = time.time() - start_time

# Evaluate the model to get loss and accuracy directly for model_2
eval_results = model_2.evaluate(X_test, y_test)
eval_loss, eval_accuracy, eval_precision, eval_recall = eval_results

# Use the model to make predictions on the test set for model_2
predictions_2 = (model_2.predict(X_test) > 0.5).astype(int)

# Calculate F1 score using predictions for model_2
eval_f1 = f1_score(y_test, predictions_2, average='binary')

# Construct the result dictionary for model_2
evaluation_results_2 = {
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,
    'eval_f1': eval_f1,
    'eval_precision': eval_precision,
    'eval_recall': eval_recall,
    'eval_runtime': training_time,
    'eval_samples_per_second': len(X_test) / training_time,
    'epoch': 20 
}

# Model 3: Type of Implicit Hate Speech
df = pd.read_csv('datasets/final_augmented_dataset.tsv', sep='\t')

df['implicit_class'] = df['implicit_class'].fillna('none')
label_encoder_imp_class = LabelEncoder()
df['implicit_class'] = label_encoder_imp_class.fit_transform(df['implicit_class'])

df['extra_implicit_class'] = df['extra_implicit_class'].fillna('none')
label_encoder_extra_imp_class = LabelEncoder()
df['extra_implicit_class'] = label_encoder_extra_imp_class.fit_transform(df['extra_implicit_class'])

df_implicit = df[df['implicit_or_explicit'] == 'implicit_hate']
tokenizer.fit_on_texts(df_implicit['post'])
sequences = tokenizer.texts_to_sequences(df_implicit['post'])
data = pad_sequences(sequences, maxlen=100)

label_encoder_imp_class = LabelEncoder()
df_implicit['implicit_class_encoded'] = label_encoder_imp_class.fit_transform(df_implicit['implicit_class'])

num_classes = df_implicit['implicit_class_encoded'].nunique()
y = df_implicit['implicit_class_encoded']
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)
model_3 = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=num_classes, activation='softmax')
])

# Modify the model compilation to include Precision and Recall metrics for model_3
model_3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Start timing your model training for model_3
start_time = time.time()
model_3.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
training_time = time.time() - start_time

# Evaluate the model to get loss and accuracy directly for model_3
eval_results = model_3.evaluate(X_test, y_test)
eval_loss, eval_accuracy, eval_precision, eval_recall = eval_results

# Use the model to make predictions on the test set for model_3
predictions_3 = model_3.predict(X_test)

# Since model_3 is for multiclass classification, adapt F1, precision, and recall calculations
predicted_classes = np.argmax(predictions_3, axis=1)
true_classes = np.argmax(y_test, axis=1)

eval_f1 = f1_score(true_classes, predicted_classes, average='macro')
eval_precision = precision_score(true_classes, predicted_classes, average='macro')
eval_recall = recall_score(true_classes, predicted_classes, average='macro')

# Construct the result dictionary for model_3
evaluation_results_3 = {
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,
    'eval_f1': eval_f1,
    'eval_precision': eval_precision,
    'eval_recall': eval_recall,
    'eval_runtime': training_time,
    'eval_samples_per_second': len(X_test) / training_time,
    'epoch': 20 
}

# Print results:
print("Evaluation Results for Hate versus Not Hate:")
print(evaluation_results_1)

print("Evaluation Results for Implicit vs Explicit:")
print(evaluation_results_2)

print("Evaluation Results for Type of Implicit HS:")
print(evaluation_results_3)
