# src/immobiliare/cli/step/run_html_tokenizer_step.py
import csv

from bs4 import BeautifulSoup

from pipeline.steps import FileLoadingStep
from preprocess.html.tokenizer.html_tokenizer import HTMLAnnuncioTokenizer
from utils import timestamped_path


def main():

    html_pages_dir_to_predict = "data/html_pages_dir/to_predict"
    output_path = "data/transformer_model/inference/ground_truth/ground_truth_generate.csv"  # 📝 Specifica qui il path di output

    file_loading = FileLoadingStep(input_dir=str(html_pages_dir_to_predict), limit=8)
    pages = file_loading.run()
    tokenizer = HTMLAnnuncioTokenizer()

    token_tuples = []

    counter = 0
    for page in pages:
        filename = page["filename"]
        content = page["content"]
        soup = BeautifulSoup(content, "html.parser")
        print(f"📄 Processing: {filename}")
        token_tuple = tokenizer.tokenize(soup)
        for tuple_val in token_tuple:
            token_text = tuple_val[0]
            #print(f"📄 Token text: {token_text}")
            token_tuples.append(token_text)
            counter += 1

    print(f"\n<UNK> Tokenization done: {counter} tokens")

    # ✨ Step di salvataggio CSV
    try:
        with open(timestamped_path(output_path), mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])  # intestazione
            for token_text in token_tuples:
                writer.writerow([token_text])
        print(f"\n✅ Token tuples salvati in: {output_path}")
    except Exception as e:
        print(f"\n❌ Errore durante il salvataggio CSV: {e}")

if __name__ == "__main__":
    main()