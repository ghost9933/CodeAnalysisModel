import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ----------------------------
# 1. Load and Preprocess Data
# ----------------------------

# Load the dataset
with open('./geeks_for_geeks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare the DataFrame
texts = []
time_labels = []
space_labels = []

for item in data:
    code_snippets = item.get('codes', {})
    if code_snippets:
        # Combine all code snippets and comments for a single item
        combined_text = ''
        for lang, code_info in code_snippets.items():
            code = code_info.get('code', '')
            comment = code_info.get('comments', '')
            combined_text += comment + '\n' + code + '\n'
        texts.append(combined_text)
        time_labels.append(item.get('time_complexity', 'Unknown'))
        space_labels.append(item.get('space_complexity', 'Unknown'))

# Encode the Labels
all_complexities = list(set(time_labels + space_labels))
complexity_encoder = LabelEncoder()
complexity_encoder.fit(all_complexities)

encoded_time_labels = complexity_encoder.transform(time_labels)
encoded_space_labels = complexity_encoder.transform(space_labels)
num_complexities = len(complexity_encoder.classes_)

# Split the Data
train_texts_t, test_texts_t, train_labels_t, test_labels_t = train_test_split(
    texts,
    encoded_time_labels,
    test_size=0.2,
    random_state=42
)

train_texts_s, test_texts_s, train_labels_s, test_labels_s = train_test_split(
    texts,
    encoded_space_labels,
    test_size=0.2,
    random_state=42
)

# Tokenize the Texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    train_texts_t,
    truncation=True,
    padding=True,
    max_length=512
)

test_encodings = tokenizer(
    test_texts_t,
    truncation=True,
    padding=True,
    max_length=512
)

# Create Dataset Objects
class CodeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels  # Labels should be numerical

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Time Complexity Dataset
train_dataset_t = CodeDataset(train_encodings, train_labels_t)
test_dataset_t = CodeDataset(test_encodings, test_labels_t)

# Space Complexity Dataset
train_dataset_s = CodeDataset(train_encodings, train_labels_s)
test_dataset_s = CodeDataset(test_encodings, test_labels_s)

# ----------------------------
# 2. Fine-tune BERT Model for Time Complexity
# ----------------------------

model_t = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_complexities
)

training_args_t = TrainingArguments(
    output_dir='./results_time',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs_time',
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=10,
)

trainer_t = Trainer(
    model=model_t,
    args=training_args_t,
    train_dataset=train_dataset_t,
    eval_dataset=test_dataset_t
)

trainer_t.train()

# Evaluate the Model
results_t = trainer_t.evaluate()
print("Time Complexity Evaluation Results:")
print(results_t)

# Save the Model
trainer_t.save_model('./saved_model_time')
tokenizer.save_pretrained('./saved_model_time')

# ----------------------------
# 3. Fine-tune BERT Model for Space Complexity
# ----------------------------

model_s = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_complexities
)

training_args_s = TrainingArguments(
    output_dir='./results_space',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs_space',
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=10,
)

trainer_s = Trainer(
    model=model_s,
    args=training_args_s,
    train_dataset=train_dataset_s,
    eval_dataset=test_dataset_s
)

trainer_s.train()

# Evaluate the Model
results_s = trainer_s.evaluate()
print("Space Complexity Evaluation Results:")
print(results_s)

# Save the Model
trainer_s.save_model('./saved_model_space')
tokenizer.save_pretrained('./saved_model_space')

# ----------------------------
# 4. Load the Trained Models for Testing
# ----------------------------

model_t = BertForSequenceClassification.from_pretrained('./saved_model_time')
model_s = BertForSequenceClassification.from_pretrained('./saved_model_space')
tokenizer = BertTokenizer.from_pretrained('./saved_model_time')
model_t.eval()
model_s.eval()

# ----------------------------
# 5. Evaluate on Test Data
# ----------------------------

# Prepare test encodings
test_encodings = tokenizer(
    test_texts_t,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors='pt'
)

# Time Complexity Predictions
with torch.no_grad():
    outputs_t = model_t(**test_encodings)
    logits_t = outputs_t.logits
    predictions_t = torch.argmax(logits_t, dim=-1).numpy()

# Space Complexity Predictions
with torch.no_grad():
    outputs_s = model_s(**test_encodings)
    logits_s = outputs_s.logits
    predictions_s = torch.argmax(logits_s, dim=-1).numpy()

# True labels
true_labels_t = np.array(test_labels_t)
true_labels_s = np.array(test_labels_s)

# Classification Reports
report_t = classification_report(true_labels_t, predictions_t, target_names=complexity_encoder.classes_)
print("Time Complexity Classification Report:")
print(report_t)

report_s = classification_report(true_labels_s, predictions_s, target_names=complexity_encoder.classes_)
print("Space Complexity Classification Report:")
print(report_s)

# ----------------------------
# 6. Make Predictions on New Inputs
# ----------------------------

custom_text = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
"""

custom_encoding = tokenizer(
    custom_text,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors='pt'
)

# Predict Time Complexity
with torch.no_grad():
    output_t = model_t(**custom_encoding)
    logits_t = output_t.logits
    predicted_class_t = torch.argmax(logits_t, dim=-1).item()
    predicted_time = complexity_encoder.inverse_transform([predicted_class_t])[0]

# Predict Space Complexity
with torch.no_grad():
    output_s = model_s(**custom_encoding)
    logits_s = output_s.logits
    predicted_class_s = torch.argmax(logits_s, dim=-1).item()
    predicted_space = complexity_encoder.inverse_transform([predicted_class_s])[0]

print(f"Predicted Time Complexity: {predicted_time}")
print(f"Predicted Space Complexity: {predicted_space}")
