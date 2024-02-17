import pandas as pd
import torch

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
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Custom function for computing evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Data
data = pd.read_csv("implicit_hate_v1_stg0_posts.tsv", delimiter='\t')

# Encode the labels
label_encoder_hate = LabelEncoder()
data['hate_or_not_hate_encoded'] = label_encoder_hate.fit_transform(data['hate_or_not_hate'])

X = data['post']  # Features for the first model
y = data['hate_or_not_hate_encoded']  # Labels for the first model

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the training and testing data
train_encodings = bert_tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
test_encodings = bert_tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")

# Create datasets
train_dataset = HateSpeechDataset(train_encodings, y_train)
test_dataset = HateSpeechDataset(test_encodings, y_test)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_hate.classes_))

training_args = TrainingArguments(
    output_dir='./results',  # Output directory for model checkpoints
    num_train_epochs=3,  # Number of epochs (adjust based on your dataset size and needs)
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Assuming you have a test_dataset
    compute_metrics=compute_metrics,  # Define your compute_metrics function for evaluation
)

trainer.train()

trainer.evaluate()

######################################### Classifier 2 #########################################

# TODO: ADD CODE HERE

######################################### Classifier 3 #########################################

# TODO: ADD CODE HERE
