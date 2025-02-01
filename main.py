from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import random

# Step 1: Load Pre-trained BanglaBERT Model and Tokenizer
model_name = "sagorsarker/bangla-bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=10)  # Adjusted for 10 labels

# Step 2: Enhanced Dataset with 500 Random Gossip Sentences
sentences = [
    "তোমার দিন কেমন কাটছে?", "আজকের আবহাওয়া কেমন?", "তুমি কি আজ রাতে ফ্রি?",
    "আমার প্রোজেক্ট ভালোভাবে চলছে।", "ক্লান্তি মাঝে মাঝে ভালো।", "তুমি আজ খুব সুন্দর লাগছো।",
    "কাল রাতের মুভি দারুণ ছিল।", "আজ কি মিটিং ঠিক সময়ে শুরু হবে?", "কফি খাবো, চলো।",
    "তোমার জন্য একটা সারপ্রাইজ আছে।"
]

responses = [
    "আমি বেশ ভালো আছি।", "আমার মনে হয় আজকের দিন ভালো যাবে।", "হ্যাঁ, আমি ফ্রি।",
    "আমার প্রোজেক্ট ঠিকঠাক চলছে।", "ক্লান্তি মাঝে মাঝে আমাদের জীবনের অংশ।",
    "ধন্যবাদ! তোমার কথায় আমি খুশি।", "হ্যাঁ, মুভিটা দারুণ ছিল।", "আমার মনে হয় মিটিং ঠিক সময়ে হবে।",
    "চলো কফি খাই।", "সারপ্রাইজের জন্য অপেক্ষা করছি।"
]

# Randomly generate additional sentences
for _ in range(490):
    gossip = f"গসিপ: {random.choice(sentences)}"
    response = random.choice(responses)
    sentences.append(gossip)
    responses.append(response)

# Encode Labels
encoder = LabelEncoder()
encoder.fit(responses)
encoded_responses = encoder.transform(responses)

# Tokenize and Encode Sentences
def encode_sentences(sentences):
    return tokenizer(sentences, truncation=True, padding=True, max_length=128, return_tensors='pt')

class BanglaDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.encodings = encode_sentences(sentences)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.sentences)

# Prepare the Dataset
train_dataset = BanglaDataset(sentences, encoded_responses)

# Step 3: Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,               # Increase epochs as needed
    per_device_train_batch_size=16,   # Adjust batch size
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",      # Match with save strategy
    save_strategy="epoch",            # Save at the end of each epoch
    save_total_limit=2,
    load_best_model_at_end=True       # Ensure the best model is loaded
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Use same dataset for simplicity
)

# Step 4: Train the Model
trainer.train()

# Step 5: Save the Model and Tokenizer
model.save_pretrained("./bangla_chatbot_model")
tokenizer.save_pretrained("./bangla_chatbot_model")

# Step 6: Chatbot Inference
model = BertForSequenceClassification.from_pretrained("./bangla_chatbot_model")
tokenizer = BertTokenizer.from_pretrained("./bangla_chatbot_model")

def chatbot_response(user_input):
    encoding = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=128)
    output = model(**encoding)
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    response = encoder.inverse_transform([predicted_label])[0]

    # Debugging outputs
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted label: {predicted_label}")

    return response

# Step 7: Test the Chatbot
while True:
    user_input = input("User: ")
    response = chatbot_response(user_input)
    print(f"Model: {response}")
