import pandas as pd
import torch
import time

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset


######################################### Classifier 1 #########################################


# Custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if torch.backends.mps.is_available():
            item = {key: val.to(torch.device("mps")) for key, val in item.items()}
        return item

    def __len__(self):
        return len(self.labels)


# Custom function for computing evaluation metrics
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

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Data
data_1 = pd.read_csv("../datasets/implicit_hate_v1_stg0_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_hate = LabelEncoder()
data_1['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data_1['class'])
data_1.reset_index(drop=True, inplace=True)

X_1 = data_1['post']  # Features for the first model
y_1 = data_1['hate_or_not_hate_encoded']  # Labels for the first model

# Split data for training and testing
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=42)
y_train_1.reset_index(drop=True, inplace=True)
y_test_1.reset_index(drop=True, inplace=True)

# Tokenize the training and testing data
train_encodings_1 = bert_tokenizer(X_train_1.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_1 = bert_tokenizer(X_test_1.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets
train_dataset_1 = HateSpeechDataset(train_encodings_1, y_train_1)
test_dataset_1 = HateSpeechDataset(test_encodings_1, y_test_1)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_hate.classes_))

# Move model to GPU if available
if torch.backends.mps.is_available():
    model.to(torch.device("mps"))

training_args = TrainingArguments(
    output_dir='./results',  # Output directory for model checkpoints
    num_train_epochs=3,  # Number of epochs (adjust based on your dataset size and needs)
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
)

trainer_1 = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_1,
    eval_dataset=test_dataset_1,
    compute_metrics=compute_metrics,
)

trainer_1.train()

# Evaluate the model
evaluation_results = trainer_1.evaluate()

# Print the evaluation metrics
print("Evaluation Results for Hate vs. Not Hate:", evaluation_results)

######################################### Classifier 2 #########################################

# Data
data_2 = pd.read_csv("../datasets/implicit_hate_v1_stg1_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_hate_type = LabelEncoder()
data_2['implicit_or_explicit_encoded'] = label_encoder_hate_type.fit_transform(data_2['class'])
data_2.reset_index(drop=True, inplace=True)

X_2 = data_2['post']  # Features for the second model
y_2 = data_2['implicit_or_explicit_encoded']  # Labels for the second model

# Split data for training and testing
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)
y_train_2.reset_index(drop=True, inplace=True)
y_test_2.reset_index(drop=True, inplace=True)

# Tokenize the training and testing data
train_encodings_2 = bert_tokenizer(X_train_2.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_2 = bert_tokenizer(X_test_2.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets
train_dataset_2 = HateSpeechDataset(train_encodings_2, y_train_2)
test_dataset_2 = HateSpeechDataset(test_encodings_2, y_test_2)

model_2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_hate_type.classes_))

# Move model to GPU if available
if torch.backends.mps.is_available():
    model_2.to(torch.device("mps"))

training_args = TrainingArguments(
    output_dir='./results_second',  # Output directory for model checkpoints
    num_train_epochs=3,  # Number of epochs (adjust based on your dataset size and needs)
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs_second',  # Directory for storing logs
)

trainer_2 = Trainer(
    model=model_2,
    args=training_args,
    train_dataset=train_dataset_2,
    eval_dataset=test_dataset_2,
    compute_metrics=compute_metrics,
)

trainer_2.train()

# Evaluate the model
evaluation_results = trainer_2.evaluate()

# Print the evaluation metrics
print("Evaluation Results for Explicit Hate vs. Implicit Hate:", evaluation_results)

######################################### Classifier 3 #########################################

# Data
data_3 = pd.read_csv("../datasets/implicit_hate_v1_stg2_posts.tsv", delimiter='\t')
# Encode the labels
label_encoder_implicit_type = LabelEncoder()
data_3['implicit_types_encoded'] = label_encoder_implicit_type.fit_transform(data_3['implicit_class'])
data_3.reset_index(drop=True, inplace=True)

X_3 = data_3['post']  # Features for the third model
y_3 = data_3['implicit_types_encoded']  # Labels for the third model

# Split data for training and testing
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=42)
y_train_3.reset_index(drop=True, inplace=True)
y_test_3.reset_index(drop=True, inplace=True)

# Tokenize the training and testing data
train_encodings_3 = bert_tokenizer(X_train_3.tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings_3 = bert_tokenizer(X_test_3.tolist(), truncation=True, padding=True, return_tensors="pt")

# Create datasets
train_dataset_3 = HateSpeechDataset(train_encodings_3, y_train_3)
test_dataset_3 = HateSpeechDataset(test_encodings_3, y_test_3)

model_3 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_implicit_type.classes_))

# Move model to GPU if available
if torch.backends.mps.is_available():
    model_3.to(torch.device("mps"))

training_args = TrainingArguments(
    output_dir='./results_third',  # Output directory for model checkpoints
    num_train_epochs=3,  # Number of epochs (adjust based on your dataset size and needs)
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs_third',  # Directory for storing logs
)

trainer_3 = Trainer(
    model=model_3,
    args=training_args,
    train_dataset=train_dataset_3,
    eval_dataset=test_dataset_3,
    compute_metrics=compute_metrics,
)

trainer_3.train()

# Evaluate the model
evaluation_results = trainer_3.evaluate()

# Print the evaluation metrics
print("Evaluation Results for Implicit Hate Types:", evaluation_results)

# Place this line at the statement you're interested in
elapsed_time = time.time() - start_time

# Calculate hours, minutes, and seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

# Format the time as a string
formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

print(f"It took {formatted_time} (hh:mm:ss) for the individual BERT classifiers.")
