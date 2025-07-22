## 1. Test con token raddoppiati  
- **Analisi di fattibilità**: portare il training da ~75 000 a ~150 000 token può ridurre ulteriormente la varianza e catturare pattern rari, ma comporta costi computazionali e rischio di diminishing returns.  
- **Vantaggi**:  
  - Migliore copertura dei casi edge (feature rare come FEATURE_LUSSO, FEATURE_DATA)  
  - Potenziale aumento di precision/recall su classi sottorappresentate  
- **Svantaggi**:  
  - Addestramento molto più lento e costoso (GPU hours ↑)  
  - Rischio di overfitting se non si aumenta proporzionalmente la diversità dei dati  
  - Diminuzione del guadagno marginale oltre una certa soglia

## 2. Miglioramento manuale del labeling  
- **Interventi manuali**:  
  - Ricontextualizzare i token in frasi complete (es. “Palazzo dotato di ascensore” vs “ascensore”)  
  - Revisionare esempi confondibili aggiungendo annotazioni contestuali  
  - Bilanciare target rare through targeted augmentation (es. generare esempi Rails per FEATURE_LUSSO)  
- **Misurazione impatto**:  
  - Confrontare metriche per classe (precision, recall, F1) prima/dopo su un validation set fisso  
  - Monitorare variazione del confusione matrix, in particolare sulle classi rare  
  - Eseguire analisi degli errori qualitativa su batch di esempi corretti manualmente  

## 3. Pipeline di labellizzazione intelligente  
- **Descrizione workflow**:  
  1. **Pre-annotazione automatica** con regole euristiche + modello attuale  
  2. **Revisione umana mirata** solo su token a bassa confidenza o classi rare  
  3. **Retraining incrementale**: aggiungere i nuovi esempi al dataset e riaddestrare il modello  
  4. **Loop di autoapprendimento**: il modello suggerisce label su nuovi dati, l’annotatore conferma solo le correzioni critiche  
- **Riscrittura concetto**:  
  > “Costruiamo un flusso ibrido in cui il modello crea bozze di etichette, un annotatore umano sistema solo gli errori ad alto rischio, e poi il modello si riaddestra in modo iterativo su queste revisioni per migliorare autonomamente le sue predizioni future.”
  
### Sequenza consigliata degli step  
1. **Step 2**: Miglioramento manuale del labeling (fissare la qualità di base)  
2. **Step 1**: Test con token raddoppiati (valutare la scala dopo aver pulito i dati)  
3. **Step 3**: Pipeline di labellizzazione intelligente (automatizzare e iterare)  

---
# Seconda risposta

## 1. Test con token raddoppiati
- **Senso dell’esperimento**  
  Dato che gli errori non crescono linearmente al crescere della dimensione (caso c con 37 420 esempi ha meno errori di b e d), estendere il training a ~150 000 token può aiutare a:
  - **Vantaggi**  
    - Ridurre la varianza sugli esempi rari (FEATURE_DATA, FEATURE_ARREDATO, FEATURE_LUSSO).  
    - Fornire più contesto per pattern semantici complessi.  
  - **Svantaggi**  
    - Costi computazionali e di memoria molto più alti (GPU, tempo di training).  
    - Diminishing returns: se il rumore di labeling è la causa principale, più dati non risolveranno l’errore di contesto.

## 2. Miglioramento manuale del labeling
- **Strategie di intervento**  
  1. **Re-annotazione contestuale**  
     - Passare da singole parole (“terrazzo”) a segmenti frasali (“Ampio terrazzo panoramico”).  
  2. **Active learning**  
     - Selezionare i token su cui il modello è insicuro (bassa confidence) e correggerli manualmente.  
  3. **Peer review**  
     - Affiancare due annotatori per i casi borderline e risolvere i disaccordi.
- **Metriche di valutazione dell’impatto**  
  - Confrontare precision, recall e F1 per ciascuna classe prima/dopo riallenamento.  
  - Monitorare la variazione del numero totale di errori di inferenza sul test set fisso.

## 3. Pipeline di labellizzazione intelligente
- **Workflow proposto**  
  1. **Inizializzazione**: addestra il modello sul dataset manuale da 75 000 esempi.  
  2. **Predizione su dati non etichettati**: usa il modello per generare label e confidence score.  
  3. **Filtraggio**: seleziona le predizioni ad alta confidenza come pseudo-label.  
  4. **Human-in-the-loop**: revisione rapida dei pseudo-label a confidenza media.  
  5. **Retraining iterativo**: unisci manual + pseudo-label e ripeti il ciclo.
- **Riscrittura concetto**  
  “Un processo semi-automatico in cui il modello etichetta nuovi dati, l’annotatore verifica solo i casi incerti, e poi si riallena in loop continuo sulle predizioni verificate.”

## Ordine consigliato di esecuzione
1. **(2) Miglioramento manuale del labeling** sul dataset da 75 000 esempi  
2. **(3) Introduzione della pipeline intelligente** per estendere velocemente il labeling  
3. **(1) Test con token raddoppiati** (~150 000) usando il dataset così ampliato  
