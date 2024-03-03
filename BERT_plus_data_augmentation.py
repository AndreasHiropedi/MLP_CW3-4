import pandas as pd
import torch
import time

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer


class SequentialHateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        # Move encodings to GPU if available
        self.encodings = {k: v.to(torch.device("mps")) for k, v in encodings.items()} if torch.backends.mps.is_available() else encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Convert labels to tensor before moving to GPU/MPS
        label_tensor = torch.tensor(self.labels[idx])
        item['labels'] = label_tensor.to(torch.device("mps")) if torch.backends.mps.is_available() else label_tensor
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Capture the start time
start_time = time.time()

# Load the dataset
data = pd.read_csv('final_augmented_dataset.tsv', delimiter='\t')

######################################### Classifier 1 #########################################

# Encode the 'hate_or_not_hate' labels
label_encoder_hate = LabelEncoder()
data['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data['hate_or_not_hate'])
data.reset_index(drop=True, inplace=True)

# For simplicity, let's focus on the first classification task: hate_or_not_hate
X = data['post']
y = data['hate_or_not_hate_encoded']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = bert_tokenizer(X_train.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings = bert_tokenizer(X_test.tolist(), truncation=True, padding=True, return_tensors="pt")

train_dataset = SequentialHateSpeechDataset(train_encodings, y_train)
test_dataset = SequentialHateSpeechDataset(test_encodings, y_test)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_hate.classes_))

# Move model to GPU if available
if torch.backends.mps.is_available():
    model.to(torch.device("mps"))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the first model on the test dataset
evaluation_results = trainer.evaluate()

# Print the evaluation metrics
print("Evaluation Results for Hate versus Not Hate:", evaluation_results)

######################################### Classifier 2 #########################################

full_encodings = bert_tokenizer(data['post'].tolist(), truncation=True, padding=True, return_tensors="pt")
full_dataset = SequentialHateSpeechDataset(full_encodings, torch.tensor(data['hate_or_not_hate_encoded'].values))

# Use Trainer.predict to get predictions for the entire dataset
predictions = trainer.predict(full_dataset)
predicted_labels = predictions.predictions.argmax(-1)

# Add these predictions back to the dataframe
data['hate_or_not_hate_pred'] = predicted_labels

# Encode the 'implicit_or_explicit' labels for the next task
label_encoder_implicit = LabelEncoder()
data['implicit_or_explicit_encoded'] = label_encoder_implicit.fit_transform(data['implicit_or_explicit'].fillna('unknown'))

# Convert numeric predictions into textual representation
# Assuming '0' for 'not hate' and '1' for 'hate'
data['hate_pred_text'] = data['hate_or_not_hate_pred'].apply(lambda x: 'HATE' if x == 1 else 'NOT_HATE')

# Concatenate this prediction text with the original post
data['post_with_pred'] = data['hate_pred_text'] + " " + data['post']
data.reset_index(drop=True, inplace=True)

# Assume 'unknown' is encoded to a specific value, find that value
unknown_label_value = label_encoder_implicit.transform(['unknown'])[0]

# Filter out rows where 'implicit_or_explicit_encoded' is not equal to the 'unknown' label value
filtered_data = data[data['implicit_or_explicit_encoded'] != unknown_label_value]

# Now, use 'filtered_data' for the rest of the steps in your second classifier
X_next = filtered_data['post_with_pred']
y_next = filtered_data['implicit_or_explicit_encoded']

# Split data for the second task
X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(X_next, y_next, test_size=0.2, random_state=42)
y_train_next.reset_index(drop=True, inplace=True)
y_test_next.reset_index(drop=True, inplace=True)

# Tokenize the data for the second task
train_encodings_next = bert_tokenizer(X_train_next.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_next = bert_tokenizer(X_test_next.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets for the second task
train_dataset_next = SequentialHateSpeechDataset(train_encodings_next, y_train_next)
test_dataset_next = SequentialHateSpeechDataset(test_encodings_next, y_test_next)

model_next = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_implicit.classes_))

# Move model to GPU if available
if torch.backends.mps.is_available():
    model_next.to(torch.device("mps"))

training_args_next = TrainingArguments(
    output_dir='./results_second',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_second',
)

trainer_next = Trainer(
    model=model_next,
    args=training_args_next,
    train_dataset=train_dataset_next,
    eval_dataset=test_dataset_next,
    compute_metrics=compute_metrics,
)

trainer_next.train()

# Evaluate the second model on the test dataset
evaluation_results_next = trainer_next.evaluate()

# Print the evaluation metrics
print("Evaluation Results for Implicit versus Explicit:", evaluation_results_next)

######################################### Classifier 3 #########################################

# Use Trainer.predict to get predictions for the entire dataset for the second task
predictions_next = trainer_next.predict(full_dataset)
predicted_labels_next = predictions_next.predictions.argmax(-1)

# Convert numeric predictions into labels
data['implicit_or_explicit_pred'] = label_encoder_implicit.inverse_transform(predicted_labels_next)

# Encode the 'implicit_class' labels for the third task
label_encoder_implicit_class = LabelEncoder()
data['implicit_class_encoded'] = label_encoder_implicit_class.fit_transform(data['implicit_class'].fillna('unknown'))

# Convert the second task's predictions into a textual representation
data['implicit_explicit_pred_text'] = data['implicit_or_explicit_pred'].apply(lambda x: str(x).upper())

# Concatenate this prediction text with the original post (already includes first task's predictions)
data['post_with_preds'] = data['post_with_pred'] + " " + data['implicit_explicit_pred_text']
data.reset_index(drop=True, inplace=True)

# Assume 'unknown' is encoded to a specific value, find that value for 'implicit_class'
unknown_class_value = label_encoder_implicit_class.transform(['unknown'])[0]

# Filter out rows where 'implicit_class_encoded' is not equal to the 'unknown' label value
filtered_data_third = data[data['implicit_class_encoded'] != unknown_class_value]

# Use 'filtered_data_third' for the rest of the steps in your third classifier
X_third = filtered_data_third['post_with_preds']
y_third = filtered_data_third['implicit_class_encoded']

# Split data for the third task
X_train_third, X_test_third, y_train_third, y_test_third = train_test_split(X_third, y_third, test_size=0.2, random_state=42)
y_train_third.reset_index(drop=True, inplace=True)
y_test_third.reset_index(drop=True, inplace=True)

# Tokenize the data for the third task
train_encodings_third = bert_tokenizer(X_train_third.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_third = bert_tokenizer(X_test_third.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets for the third task
train_dataset_third = SequentialHateSpeechDataset(train_encodings_third, y_train_third)
test_dataset_third = SequentialHateSpeechDataset(test_encodings_third, y_test_third)

model_third = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_implicit_class.classes_))

# Move model to GPU if available
if torch.backends.mps.is_available():
    model_third.to(torch.device("mps"))

training_args_third = TrainingArguments(
    output_dir='./results_third',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_third',
)

trainer_third = Trainer(
    model=model_third,
    args=training_args_third,
    train_dataset=train_dataset_third,
    eval_dataset=test_dataset_third,
    compute_metrics=compute_metrics,  # Ensure this function is adapted for the third task
)

trainer_third.train()

# Evaluate the third model on the test dataset
evaluation_results_third = trainer_third.evaluate()

# Print the evaluation metrics
print("Evaluation Results for Implicit Hate Types:", evaluation_results_third)

# Place this line at the statement you're interested in
elapsed_time = time.time() - start_time

# Calculate hours, minutes, and seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

# Format the time as a string
formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

print(f"It took {formatted_time} (hh:mm:ss) for the chained BERT classifier.")
