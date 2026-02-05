from datasets import load_dataset
import os
import itertools

LANGUAGES = {
    "en": "English",
    "ru": "Russian",
    "zh": "Mandarin"
}

# target num of charactes per language
TARGET_CHARS_PER_LANG = 30_000_000
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_corpus.txt")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def stream_text_from_dataset(dataset_name, dataset_config, text_field="text"):
    ds = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True
    )
    for row in ds:
        if text_field in row and row[text_field]:
            yield normalize_text(row[text_field])

def collect_chars(generator, target_chars):
    collected = []
    total_chars = 0

    for text in generator:
        collected.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break

    print(f"Collected {total_chars:,} characters")
    return "".join(collected)

## main 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    combined_corpus = []

    for lang in LANGUAGES:
        print(f"\n Processing language: {LANGUAGES[lang]} ({lang}) ===")

        lang_texts = []

        # oscar - not accessible currently, requested access 
        try:
            print("Loading OSCAR...")
            oscar_gen = stream_text_from_dataset(
                "oscar-corpus/oscar",
                lang
            )
            lang_texts.append(
                collect_chars(oscar_gen, TARGET_CHARS_PER_LANG // 2)
            )
        except Exception as e:
            print(f"Skipping OSCAR for {lang} (gated or unavailable): {e}")


        # wikipedia multilingual
        print("Loading Wikipedia...")
        wiki_gen = stream_text_from_dataset(
            "Cohere/wikipedia-2023-11-embed-multilingual-v3",
            lang
        )
        lang_texts.append(
            collect_chars(wiki_gen, TARGET_CHARS_PER_LANG // 4)
        )

        # project gutenberg
        try:
            print("Loading Project Gutenberg...")
            gutenberg_gen = stream_text_from_dataset(
                "manu/project_gutenberg",
                lang
            )
            lang_texts.append(
                collect_chars(gutenberg_gen, TARGET_CHARS_PER_LANG // 4)
            )
        except Exception as e:
            print(f"Skipping Project Gutenberg for {lang}: {e}")

        language_block = "".join(lang_texts)
        print(f"Total for {lang}: {len(language_block):,} characters")

        combined_corpus.append(language_block)

    final_text = "".join(combined_corpus)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_text)

    print("\n=== DONE ===")
    print(f"Final corpus size: {len(final_text):,} characters")
    print(f"Saved to: {OUTPUT_FILE}")

    os._exit(0)

if __name__ == "__main__":
    main()