with open("src/data/train_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

split_index = int(len(text) * 0.9)

eval_text = text[split_index:]

with open("src/data/eval_corpus.txt", "w", encoding="utf-8") as f:
    f.write(eval_text)

print("Eval corpus created.")