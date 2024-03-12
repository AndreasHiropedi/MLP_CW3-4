import pandas as pd
import time
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class HierarchicalClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes_stage2=2, num_classes_stage3=7):
        super(HierarchicalClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        bert_output_size = self.bert.config.hidden_size
        self.stage1_classifier = nn.Linear(bert_output_size, 2)  # 2 classes for stage 1

        # Embedding layers for predictions to bring them to the same dimensional space as BERT output
        self.prediction_embedding_stage1 = nn.Embedding(2, bert_output_size)  # 2 classes from stage 1
        self.prediction_embedding_stage2 = nn.Embedding(2, bert_output_size)  # 2 classes from stage 2

        # Classifiers for stage 2 and 3 after prediction embedding concatenation
        self.stage2_classifier = nn.Linear(bert_output_size * 2, num_classes_stage2)  # 2 classes for stage 2
        self.stage3_classifier = nn.Linear(bert_output_size * 2, num_classes_stage3)  # 7 classes for stage 3

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # BERT processing
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Stage 1 classification
        stage1_output = self.stage1_classifier(pooled_output)

        # Stage 1 prediction to binary feature and embed
        stage1_pred = torch.argmax(stage1_output, dim=1)
        stage1_feature_embedded = self.prediction_embedding_stage1(stage1_pred)

        # Concatenate for stage 2 input
        concatenated_features_stage2 = torch.cat((pooled_output, stage1_feature_embedded), dim=1)
        stage2_output = self.stage2_classifier(self.dropout(concatenated_features_stage2))

        # Stage 2 prediction to binary feature and embed for stage 3
        stage2_pred = torch.argmax(stage2_output, dim=1)
        stage2_feature_embedded = self.prediction_embedding_stage2(stage2_pred)

        # Concatenate for stage 3 input
        concatenated_features_stage3 = torch.cat((pooled_output, stage2_feature_embedded), dim=1)
        stage3_output = self.stage3_classifier(self.dropout(concatenated_features_stage3))

        return stage1_output, stage2_output, stage3_output


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels_stage1, labels_stage2, labels_stage3):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        self.labels_stage1 = labels_stage1
        self.labels_stage2 = labels_stage2
        self.labels_stage3 = labels_stage3

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels_stage1'] = torch.tensor(self.labels_stage1[idx], dtype=torch.long)
        item['labels_stage2'] = torch.tensor(self.labels_stage2[idx], dtype=torch.long)
        item['labels_stage3'] = torch.tensor(self.labels_stage3[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels_stage1)


def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    true_labels_stage1, predictions_stage1 = [], []
    true_labels_stage2, predictions_stage2 = [], []
    true_labels_stage3, predictions_stage3 = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Note: Keep labels on CPU for metric calculations
            labels_stage1 = batch['labels_stage1'].numpy()  # These are used for metrics calculation, stay on CPU
            labels_stage2 = batch['labels_stage2'].numpy()
            labels_stage3 = batch['labels_stage3'].numpy()

            outputs = model(input_ids, attention_mask)
            pred_labels_stage1 = torch.argmax(outputs[0], dim=1).cpu().numpy()
            pred_labels_stage2 = torch.argmax(outputs[1], dim=1).cpu().numpy()
            pred_labels_stage3 = torch.argmax(outputs[2], dim=1).cpu().numpy()

            true_labels_stage1.extend(labels_stage1)
            predictions_stage1.extend(pred_labels_stage1)
            true_labels_stage2.extend(labels_stage2)
            predictions_stage2.extend(pred_labels_stage2)
            true_labels_stage3.extend(labels_stage3)
            predictions_stage3.extend(pred_labels_stage3)

    metrics = {
        'Stage 1': {
            'Accuracy': accuracy_score(true_labels_stage1, predictions_stage1),
            'F1': f1_score(true_labels_stage1, predictions_stage1, average='weighted'),
            'Precision': precision_score(true_labels_stage1, predictions_stage1, average='weighted'),
            'Recall': recall_score(true_labels_stage1, predictions_stage1, average='weighted'),
        },
        'Stage 2': {
            'Accuracy': accuracy_score(true_labels_stage2, predictions_stage2),
            'F1': f1_score(true_labels_stage2, predictions_stage2, average='weighted'),
            'Precision': precision_score(true_labels_stage2, predictions_stage2, average='weighted'),
            'Recall': recall_score(true_labels_stage2, predictions_stage2, average='weighted'),
        },
        'Stage 3': {
            'Accuracy': accuracy_score(true_labels_stage3, predictions_stage3),
            'F1': f1_score(true_labels_stage3, predictions_stage3, average='weighted'),
            'Precision': precision_score(true_labels_stage3, predictions_stage3, average='weighted'),
            'Recall': recall_score(true_labels_stage3, predictions_stage3, average='weighted'),
        }
    }

    return metrics


# Capture the start time
start_time = time.time()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
data = pd.read_csv('../datasets/final_augmented_dataset.tsv', sep='\t')

# Stage 1 does not depend on previous stages, so you can directly encode
le_stage1 = LabelEncoder()
non_nan_stage1 = data[~data['hate_or_not_hate'].isna()]  # Exclude NaN values for stage 1
data.loc[~data['hate_or_not_hate'].isna(), 'labels_stage1'] = le_stage1.fit_transform(non_nan_stage1['hate_or_not_hate'])

# For stages 2 and 3, follow a similar approach
le_stage2 = LabelEncoder()
non_nan_stage2 = data[~data['implicit_or_explicit'].isna()]  # Exclude NaN values for stage 2
data.loc[~data['implicit_or_explicit'].isna(), 'labels_stage2'] = le_stage2.fit_transform(non_nan_stage2['implicit_or_explicit'])

le_stage3 = LabelEncoder()
non_nan_stage3 = data[~data['implicit_class'].isna()]  # Exclude NaN values for stage 3
data.loc[~data['implicit_class'].isna(), 'labels_stage3'] = le_stage3.fit_transform(non_nan_stage3['implicit_class'])

# Splitting assuming label columns are prepared as 'labels_stage1', 'labels_stage2', and 'labels_stage3'
train_data, val_data = train_test_split(data.dropna(subset=['labels_stage1', 'labels_stage2', 'labels_stage3']), test_size=0.2, random_state=42)

# Now, proceed to create your datasets
train_dataset = TextDataset(tokenizer, train_data['post'].tolist(), train_data['labels_stage1'].tolist(), train_data['labels_stage2'].tolist(), train_data['labels_stage3'].tolist())
val_dataset = TextDataset(tokenizer, val_data['post'].tolist(), val_data['labels_stage1'].tolist(), val_data['labels_stage2'].tolist(), val_data['labels_stage3'].tolist())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = HierarchicalClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 20

# Create a loss function instance outside the loop to avoid recreating it every time
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()  # Make sure the model is in training mode
    total_loss = 0  # To track the loss across batches
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_stage1 = batch['labels_stage1'].to(device)
        labels_stage2 = batch['labels_stage2'].to(device)
        labels_stage3 = batch['labels_stage3'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        loss1 = loss_fn(outputs[0], labels_stage1)
        loss2 = loss_fn(outputs[1], labels_stage2)
        loss3 = loss_fn(outputs[2], labels_stage3)

        # Combine losses and perform backward pass
        total_loss_batch = loss1 + loss2 + loss3
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()  # `.item()` to get the scalar value

    # After each epoch, you might want to print the average loss
    avg_loss = total_loss / len(train_loader)
    print(f"Average loss after epoch {epoch + 1}: {avg_loss}")

metrics = evaluate_model(model, val_loader)
for stage, stage_metrics in metrics.items():
    print(f"\nMetrics for {stage}:")
    for metric_name, metric_value in stage_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

# Place this line at the statement you're interested in
elapsed_time = time.time() - start_time

# Calculate hours, minutes, and seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

# Format the time as a string
formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

print(f"It took {formatted_time} (hh:mm:ss) for the chained hybrid classifier.")
