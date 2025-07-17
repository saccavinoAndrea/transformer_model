# Riassunto Progetto: HTML Scraper con Machine Learning (Andrea)

---

## 1. Obiettivo del Progetto
- Creare uno **scraper HTML intelligente** per estrarre automaticamente informazioni strutturate da pagine web (come prezzo, descrizione, numero di locali, superficie, ecc.).
- Utilizzo di tecniche **Machine Learning supervisionato** combinate con regole manuali.
- Alta **efficienza** e **scalabilità**, anche su grandi dataset.

---

## 2. Tecnologie Utilizzate
- **Python** (ambiente virtuale attivo)
- Parsing HTML: `BeautifulSoup`
- Scraping asincrono: `Playwright`
- Machine learning: `scikit-learn`
- Analisi e salvataggio dati: `pandas`, `joblib`, `json`
- Profilazione performance: `cProfile`, `Snakeviz`

---

## 3. Pipeline Principale
1. Download HTML con fallback su errori DNS o rete  
2. Parsing con BeautifulSoup  
3. Tokenizzazione DOM (estrazione dei nodi testuali significativi)  
4. Calcolo di ~100 feature per token  
5. Classificazione token con modello ML (tipo `RandomForestClassifier`)  
6. Ricostruzione dell’output in formato JSON con le informazioni etichettate  

---

## 4. Feature Extraction
- Implementazione del modulo `FeatureExtractorOptimized`, composto da varie classi modulari (`DensityFeaturesOptimized`, `TagFeatures`, `TextFeatures`, etc.).
- Le feature includono:
  - Lunghezza testo, presenza di numeri, tag HTML, profondità nel DOM
  - Densità di testo, rapporto testo/tag, posizione nel blocco, ecc.
- Le prestazioni sono state **notevolmente migliorate**:
  - Inizialmente `DensityFeaturesOptimized` richiamava ~25 milioni di volte funzioni lente
  - Ottimizzato caching e mapping tag→int
  - Obiettivo raggiunto: **< 1 secondo per pagina**

---

## 5. Architettura Software
- Moduli organizzati per responsabilità:
  - `feature_extractor.py`: coordina le varie sottoclassi di estrazione feature  
  - `dom_tokenizer.py`: crea i token da tag/nodi HTML  
  - `token_classifier.py`: gestisce il modello e le predizioni  
  - `html_scraper.py`: orchestratore end-to-end del processo  
- Uso di interfacce astratte (`IFeatureExtractor`, `IFeature`, `ITokenClassifier`) per rendere il codice testabile e manutenibile

---

## 6. Etichettatura e Addestramento
- Pipeline completa:
  - Parsing HTML → feature extraction → salvataggio training set  
- Dataset salvato in `.jsonl` con embedding token + label associata  
- Gestione mapping `label2id` e `feature_keys_to_use` caricati/salvati da JSON  
- Funzione per filtrare classi rare (`_filter_rare_classes`)

---

## 7. Performance e Profilazione
- Analisi dei colli di bottiglia con `cProfile`  
- Inizialmente le performance erano critiche (25 milioni di chiamate in 25s)  
- Interventi chiave:
  - Pre-mappatura dei tag HTML a interi  
  - Eliminazione di ridondanze nei cicli  
  - Riduzione delle inizializzazioni ripetute  
- Risultato: **estrazione di tutte le feature sotto 1 secondo per pagina**

---

## 8. Approfondimenti ML
- Discussione su strategie future:
  - ✅ **Semi-supervised learning**: per sfruttare anche HTML non etichettati  
  - 🧠 **Reinforcement learning**: valutato, ma per ora non applicato  
  - ⚙️ Valutazione di ranking delle predizioni (es. ordinamento dei token rilevanti)  
- Addestramento supervisionato classico per ora

---

## 9. Future Roadmap
- Costruzione interfaccia per **labeling semi-automatico**  
- Possibile implementazione di un **active learning loop**  
- Generalizzazione a più domini HTML (non solo immobiliare)  
- Potenziale deploy come microservizio scalabile  
- Logging + versionamento dei modelli e delle feature  

---

## 10. Note Aggiuntive
- Le feature originali non dovevano essere cambiate: sono state mantenute tutte, anche in versione ottimizzata  
- Alcune modifiche di performance iniziali avevano cambiato i risultati → ripristinate le logiche originali con attenzione  
- Feature opzionali gestite con flessibilità nei dizionari dei token  
- Separazione della logica di scraping, estrazione e ML in file e moduli dedicati  

---

