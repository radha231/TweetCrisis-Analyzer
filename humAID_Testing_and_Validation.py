# humAID with testing and dev(validations)
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os
import json

data_folder = "/content/drive/MyDrive/humAID/"

def load_all_splits(base_folder):
    train_list, dev_list, test_list = [], [], []
    for subfolder in os.listdir(base_folder):
        subpath = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subpath):  # skip non-folders
            continue

        train_file = os.path.join(subpath, f"{subfolder}_train.tsv")
        dev_file = os.path.join(subpath, f"{subfolder}_dev.tsv")
        test_file = os.path.join(subpath, f"{subfolder}_test.tsv")

        if os.path.exists(train_file):
            df_train = pd.read_csv(train_file, sep='\t', on_bad_lines='skip')
            df_train['disaster_type'] = subfolder
            train_list.append(df_train)
        if os.path.exists(dev_file):
            df_dev = pd.read_csv(dev_file, sep='\t', on_bad_lines='skip')
            df_dev['disaster_type'] = subfolder
            dev_list.append(df_dev)
        if os.path.exists(test_file):
            df_test = pd.read_csv(test_file, sep='\t', on_bad_lines='skip')
            df_test['disaster_type'] = subfolder
            test_list.append(df_test)

    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    dev_df = pd.concat(dev_list, ignore_index=True) if dev_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    print(f"Loaded: {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test samples.")
    return train_df, dev_df, test_df

train_df, dev_df, test_df = load_all_splits(data_folder)

for df in [train_df, dev_df, test_df]:
    df.rename(columns={'tweet_text': 'text', 'class_label': 'text_humanitarian'}, inplace=True)
    df.dropna(subset=['text', 'text_humanitarian'], inplace=True)

# Combine train + dev for final training
train_df = pd.concat([train_df, dev_df], ignore_index=True)

disaster_encoder = LabelEncoder()
human_encoder = LabelEncoder()

train_df['disaster_label'] = disaster_encoder.fit_transform(train_df['disaster_type'])
train_df['human_label'] = human_encoder.fit_transform(train_df['text_humanitarian'])
test_df['disaster_label'] = disaster_encoder.transform(test_df['disaster_type'])
test_df['human_label'] = human_encoder.transform(test_df['text_humanitarian'])

num_labels_disaster = len(disaster_encoder.classes_)
num_labels_human = len(human_encoder.classes_)

print("\nDisaster types:", list(disaster_encoder.classes_))
print("Humanitarian types:", list(human_encoder.classes_))

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

class CrisisDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.disaster = df['disaster_label'].tolist()
        self.human = df['human_label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'disaster_label': torch.tensor(self.disaster[idx]),
            'human_label': torch.tensor(self.human[idx])
        }

train_dataset = CrisisDataset(train_df)
test_dataset = CrisisDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class BertweetMultiHead(nn.Module):
    def __init__(self, base_model_name, num_labels_disaster, num_labels_human):
        super().__init__()
        self.bertweet = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.bertweet.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.disaster_head = nn.Linear(hidden_size, num_labels_disaster)
        self.human_head = nn.Linear(hidden_size, num_labels_human)

    def forward(self, input_ids, attention_mask):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.disaster_head(pooled), self.human_head(pooled)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertweetMultiHead("vinai/bertweet-base", num_labels_disaster, num_labels_human).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
epochs = 5
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        disaster_label = batch['disaster_label'].to(device)
        human_label = batch['human_label'].to(device)

        d_logits, h_logits = model(input_ids, attention_mask)
        loss = criterion(d_logits, disaster_label) + criterion(h_logits, human_label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(train_loader):.4f}")

model.eval()
results = []
true_disaster_all, pred_disaster_all = [], []
true_human_all, pred_human_all = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        disaster_label = batch['disaster_label'].to(device)
        human_label = batch['human_label'].to(device)

        d_logits, h_logits = model(input_ids, attention_mask)
        d_preds = torch.argmax(d_logits, axis=1)
        h_preds = torch.argmax(h_logits, axis=1)

        true_disaster_all.extend(disaster_label.cpu().tolist())
        pred_disaster_all.extend(d_preds.cpu().tolist())
        true_human_all.extend(human_label.cpu().tolist())
        pred_human_all.extend(h_preds.cpu().tolist())

        for i in range(len(d_preds)):
            results.append({
                "text": tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                "true_disaster": disaster_encoder.inverse_transform([disaster_label[i].cpu().item()])[0],
                "pred_disaster": disaster_encoder.inverse_transform([d_preds[i].cpu().item()])[0],
                "true_human": human_encoder.inverse_transform([human_label[i].cpu().item()])[0],
                "pred_human": human_encoder.inverse_transform([h_preds[i].cpu().item()])[0]
            })

results_df = pd.DataFrame(results)
results_csv_path = "/content/drive/MyDrive/humAID_test_predictions.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\n📁 Saved full test predictions to: {results_csv_path}")

print("\n===== Disaster Type Classification =====")
print(classification_report(true_disaster_all, pred_disaster_all, target_names=disaster_encoder.classes_))
print("Accuracy:", accuracy_score(true_disaster_all, pred_disaster_all))

print("\n===== Humanitarian Type Classification =====")
print(classification_report(true_human_all, pred_human_all, target_names=human_encoder.classes_))
print("Accuracy:", accuracy_score(true_human_all, pred_human_all))

save_path = "/content/drive/MyDrive/humAID_BERTweet_Model"
os.makedirs(save_path, exist_ok=True)

torch.save(model.state_dict(), f"{save_path}/bertweet_multitask.pth")
tokenizer.save_pretrained(save_path)

with open(f"{save_path}/label_maps.json", "w") as f:
    json.dump({
        "disaster_labels": dict(enumerate(disaster_encoder.classes_)),
        "human_labels": dict(enumerate(human_encoder.classes_))
    }, f)

print(f"\n✅ Model, tokenizer, and label maps saved to: {save_path}")
