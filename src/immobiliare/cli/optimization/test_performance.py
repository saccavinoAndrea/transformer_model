import cProfile
import pstats
from pathlib import Path
import sys
from typing import List

from bs4 import BeautifulSoup

from immobiliare.domain import Token
from immobiliare.preprocess.html.feature.feature_extractor import FeatureExtractor
from immobiliare.preprocess.html.feature.feature_extractor_new import FeatureExtractorOptimized
from immobiliare.preprocess.html.tokenizer.html_tokenizer import HTMLAnnuncioTokenizer

# Assicurati che il path del tuo modulo sia accessibile
sys.path.append(str(Path(__file__).resolve().parent))


# Carica un esempio di HTML
html = Path("data/html_pages_dir/annuncio_for_training_1.html").read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")
tokenizer = HTMLAnnuncioTokenizer()
# Ottieni i token
token_tuples = tokenizer.tokenize(soup)  # deve restituire List[Tuple[str, Tag]]

def run():

    extractor = FeatureExtractorOptimized ()
    features: List[Token] = extractor.process_all(token_tuples)

    for tok in features[1:2]:
        #print(tok)  # mostrerà Token(text=..., length=..., n_words=..., …)
        print(tok.to_dict())  # o dict(tok) per vedere tutte le feature

if __name__ == "__main__":
    cProfile.run('run()', filename='profiling.prof')
    stats = pstats.Stats('profiling.prof')
    stats.strip_dirs().sort_stats("cumtime").print_stats(30)
