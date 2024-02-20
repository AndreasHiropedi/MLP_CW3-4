import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer


class SequentialHateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
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


# Load the dataset
data = pd.read_csv('implicit_hate_v1_stg0-2_posts.tsv', delimiter='\t')

######################################### Classifier 1 #########################################

# Encode the 'hate_or_not_hate' labels
label_encoder_hate = LabelEncoder()
data['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data['hate_or_not_hate'])

# For simplicity, let's focus on the first classification task: hate_or_not_hate
X = data['post']
y = data['hate_or_not_hate_encoded']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = bert_tokenizer(X_train.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings = bert_tokenizer(X_test.tolist(), truncation=True, padding=True, return_tensors="pt")

train_dataset = SequentialHateSpeechDataset(train_encodings, y_train)
test_dataset = SequentialHateSpeechDataset(test_encodings, y_test)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_hate.classes_))

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

X_next = data['post_with_pred']
y_next = data['implicit_or_explicit_encoded']

# Split data for the second task
X_train_next, X_test_next, y_train_next, y_test_next = train_test_split(X_next, y_next, test_size=0.2, random_state=42)

# Tokenize the data for the second task
train_encodings_next = bert_tokenizer(X_train_next.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_next = bert_tokenizer(X_test_next.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets for the second task
train_dataset_next = SequentialHateSpeechDataset(train_encodings_next, y_train_next)
test_dataset_next = SequentialHateSpeechDataset(test_encodings_next, y_test_next)

model_next = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_implicit.classes_))

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

X_third = data['post_with_preds']
y_third = data['implicit_class_encoded']

# Split data for the third task
X_train_third, X_test_third, y_train_third, y_test_third = train_test_split(X_third, y_third, test_size=0.2, random_state=42)

# Tokenize the data for the third task
train_encodings_third = bert_tokenizer(X_train_third.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_third = bert_tokenizer(X_test_third.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets for the third task
train_dataset_third = SequentialHateSpeechDataset(train_encodings_third, y_train_third)
test_dataset_third = SequentialHateSpeechDataset(test_encodings_third, y_test_third)

model_third = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_implicit_class.classes_))

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
