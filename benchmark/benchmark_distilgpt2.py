import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

max_len = model.config.n_positions  # 1024

total_correct = 0
total_predictions = 0
total_time = 0

with open("src/data/train_corpus.txt", "r", encoding="utf-8") as f:
    while True:
        text_chunk = f.read(4000)
        if not text_chunk:
            break

        # Force tokenizer to respect max_len
        encodings = tokenizer(
            text_chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        )

        input_ids = encodings["input_ids"][0]

        if len(input_ids) < 2:
            continue

        input_ids = input_ids.unsqueeze(0)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        total_time += (time.time() - start_time)

        for i in range(1, input_ids.shape[1]):
            true_token = input_ids[0, i]
            position_logits = logits[0, i - 1]
            top3_ids = torch.topk(position_logits, 3).indices

            if true_token in top3_ids:
                total_correct += 1

            total_predictions += 1

top3_acc = total_correct / total_predictions
error_rate = 1 - top3_acc
avg_ms = (total_time / total_predictions) * 1000

print("===================================")
print(f"Total Predictions: {total_predictions}")
print(f"Top-3 Accuracy: {top3_acc:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Average ms per prediction: {avg_ms:.4f}")
print("===================================")
