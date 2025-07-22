# 🧠 Analisi Modello Transformer PyTorch – Parte 1

## 1. Obiettivo del Progetto 
   - Valutare l’efficacia di un modello Transformer per token classification scritto in PyTorch.

   - Classificare token HTML in categorie come FEATURE_PREZZO, FEATURE_TITOLO, FEATURE_SUPERFICIE, ecc.

   - Eseguire inferenze su dataset di dimensione crescente, analizzando l'andamento degli errori.

   - Identificare pattern negli errori e possibili cause (es. problemi di labeling, squilibrio delle classi, overfitting, ecc.).
---

## 2. Risultati Inferenza: Errori vs Dataset
| Dataset | # Esempi | # Feature | Errori Predetti |
| ------- | -------- | --------- | --------------- |
| a       | 15.350   | 100       | 26              |
| b       | 31.020   | 100       | 33              |
| c       | 37.420   | 100       | 18              |
| d       | 72.964   | 100       | 32              |

🟢 Nota: Il numero di errori non cresce proporzionalmente alla dimensione del dataset. Migliori prestazioni rilevate nel caso (c).
---

## 3. Distribuzione delle Classi nei Dataset
Le etichette sono fortemente sbilanciate verso la classe O (non-feature). Le feature più rappresentate sono:
FEATURE_PREZZO, FEATURE_SUPERFICIE, FEATURE_LOCALI, FEATURE_TITOLO, FEATURE_ASCENSORE, FEATURE_BAGNI, FEATURE_PIANO.

Esempio per il caso d:

```yaml
O: 50179
FEATURE_PREZZO: 2772
FEATURE_SUPERFICIE: 2760
FEATURE_ASCENSORE: 2505
FEATURE_TITOLO: 2500
FEATURE_LOCALI: 2494
FEATURE_BAGNI: 2409
...
FEATURE_DATA: 3
```
---

## 4. Analisi Errori di Classificazione
| Classe Predetta     | Errori Comuni                                              |
| ------------------- | ---------------------------------------------------------- |
| `FEATURE_ARREDATO`  | Frasi come “Cucina arredata”, “giardino”                   |
| `FEATURE_TITOLO`    | Frasi iniziali generiche dell’annuncio                     |
| `FEATURE_ASCENSORE` | Frasi su “piani intermedi”, “piscina”                      |
| `FEATURE_CANTINA`   | Frasi su “garage”, “case da ristrutturare”                 |
| `FEATURE_TERRAZZO`  | Frasi con “piscina”, “terrazzo”, o addirittura prezzi (!?) |
| `FEATURE_PREZZO`    | Confusione con “Nomi di agenzie immobiliari”               |
-🔍 Errori più frequenti per classe (stimati da output CSV)
⚠️ Alcuni errori sembrano fuori contesto semantico, es. piscina predetta come FEATURE_ASCENSORE.
---

## 5. Osservazioni sull’Etiichettatura Heuristica
```csv
FEATURE_PREZZO,€ 860.000
FEATURE_TITOLO,"Appartamento via Francesco Ferrara..."
FEATURE_LOCALI,5+ locali
FEATURE_SUPERFICIE,180 m²
FEATURE_BAGNI,3 bagni
FEATURE_PIANO,Piano 1
FEATURE_ASCENSORE,Ascensore
FEATURE_BALCONE,Balcone
FEATURE_CANTINA,Cantina
```
📌 In questo caso, le parole etichettate si trovano nel contesto informativo completo. Il modello può apprendere relazioni corrette.

❌ Esempio di Etichettatura Fuorviante
```csv
FEATURE_ASCENSORE,ascensore
FEATURE_CANTINA,cantina
FEATURE_TERRAZZO,terrazzo
```

🔴 Questo labeling euristico è troppo minimale, privo di contesto.
Rende difficile per il modello generalizzare correttamente → porta a falsi positivi/negativi.

---
## 6. Ipotesi sull'origine degli errori
   - Etichettatura parziale o ambigua → fonte principale di errore.
   - Alcune parole possono essere correttamente predette solo nel giusto contesto semantico.
   - Il modello apprende pattern sbagliati da frasi isolate (es. “terrazzo” ≠ FEATURE_TERRAZZO sempre).

---
## 7. Distribuzione delle Classi nei Dataset Annotati
| Caso | `O` (background) | `FEATURE_PREZZO` | `FEATURE_SUPERFICIE` | `FEATURE_TITOLO` | `FEATURE_LOCALI` | `FEATURE_ASCENSORE` | `FEATURE_BAGNI` | `FEATURE_PIANO` | `FEATURE_CANTINA` | `FEATURE_BALCONE` | `FEATURE_TERRAZZO` | `FEATURE_ARREDATO` | `FEATURE_LUSSO` | `FEATURE_DATA` |
| ---- | ---------------- | ---------------- | -------------------- | ---------------- | ---------------- | ------------------- | --------------- | --------------- | ----------------- | ----------------- | ------------------ | ------------------ | --------------- | -------------- |
| a    | 10,496           | 583              | 603                  | 500              | 500              | 493                 | 477             | 441             | 307               | 265               | 215                | 176                | 179             | -              |
| b    | 21,826           | 1178             | 1220                 | 1000             | 999              | 982                 | 950             | 810             | 363               | 541               | 447                | 214                | 285             | -              |
| c    | 25,688           | 1422             | 1440                 | 1250             | 1248             | 1244                | 1194            | 1076            | 886               | 841               | 455                | 362                | 313             | -              |
| d    | 50,179           | 2772             | 2760                 | 2500             | 2494             | 2505                | 2409            | 2096            | 1521              | 1574              | 948                | 692                | 510             | 3              |

📌 Osservazioni:
   - Le classi sono molto sbilanciate: il label O è sempre dominante (>80%).
   - Le classi FEATURE_DATA, FEATURE_ARREDATO, FEATURE_LUSSO sono molto rare → rischio alto di underfitting o ignoranza da parte del modello.
   - La quantità di dati aumenta gradualmente da a → d, rendendo d il più robusto per l’addestramento.

---
## 8. Suggerimenti per Miglioramento Fase di Labeling
### 🔧 Strategie di correzione
   - Non etichettare singole parole isolate se non sono semanticamente informative da sole.
   - Includere più contesto token-based nel labeling (frasi complete, descrizioni).
   - Usare regole euristiche + regole di contesto per assegnare i label in modo più robusto.
### 💡 Esempi di trasformazione
| Labeling Minimalista           | Labeling Migliorato                                             |
| ------------------------------ | --------------------------------------------------------------- |
| `FEATURE_CANTINA`, cantina     | `FEATURE_CANTINA`, **"Presente cantina al piano seminterrato"** |
| `FEATURE_TERRAZZO`, terrazzo   | `FEATURE_TERRAZZO`, **"Ampio terrazzo panoramico"**             |
| `FEATURE_ASCENSORE`, ascensore | `FEATURE_ASCENSORE`, **"L’edificio dispone di ascensore"**      |

---
## 9. Osservazioni Specifiche sull’Errore di Classificazione
### 🔍 Analisi qualitativa degli errori frequenti (da CSV)
| Classe Predetta     | Frasi / Token spesso erroneamente classificati |
| ------------------- | ---------------------------------------------- |
| `FEATURE_ARREDATO`  | "Cucina arredata", "giardino"                  |
| `FEATURE_TITOLO`    | Frasi iniziali generiche degli annunci         |
| `FEATURE_ASCENSORE` | "piani intermedi", "piscina" (!), "ascensore"  |
| `FEATURE_CANTINA`   | "garage", "case da ristrutturare"              |
| `FEATURE_TERRAZZO`  | "piscina", "terrazzo", "prezzi case" (!?)      |
| `FEATURE_PREZZO`    | Nomi di agenzie immobiliari                    |

### 🧠 Osservazione Utente sul Labeling
    Il modello sbaglia perché il labeling è troppo minimalista, con singole parole isolate (es. “terrazzo”, “cantina”, “ascensore”) che non danno contesto sufficiente.
#### ✅ Buon esempio di labeling
    FEATURE_CANTINA,Cantina
    FEATURE_TERRAZZO,Terrazzo
    FEATURE_ASCENSORE,Ascensore
#### ➡️ NO: troppo povero, non contestualizza.
    FEATURE_CANTINA,"Presente una comoda cantina al piano seminterrato"
    FEATURE_TERRAZZO,"Terrazzo ampio con vista panoramica"
    FEATURE_ASCENSORE,"Palazzo dotato di ascensore"
➡️ SÌ: frase completa, semantica chiara e utile al modello.

---
## 10. Implicazioni sui Risultati
   - Gli errori di labeling causano ambiguità nel significato durante l’addestramento.
   - Alcuni token etichettati vengono replicati in contesti diversi → confusione per il modello (es. “cantina” presente in contesti garage o “da ristrutturare”).
   - Errori ricorrenti su classi con semantica indiretta o implicita (es. FEATURE_LUSSO, FEATURE_ARREDATO).

---
## 11. Suggerimenti Futuri

### 🔁 Miglioramento Dataset
   - Introdurre una pipeline semi-automatica per verifica e revisione dei labeling.
   - Utilizzare modelli linguistici (come GPT) per suggerire correzioni o label alternativi.
   - Controlli di coerenza semantica fra classi simili.
### 📊 Valutazione Modello
   - Introdurre metriche per classe (precision, recall, F1 per ogni FEATURE_*).
   - Valutare errori in base alla vicinanza semantica (es. cantina vs garage meno grave di cantina vs ascensore).
   - Isolare gli esempi più confondibili per l’analisi manuale.

---
## 12. Distribuzione delle Classi nei 4 Dataset
##### Tabella comparativa delle distribuzioni LABEL nei casi a, b, c, d:
| Classe               | Caso A | Caso B | Caso C | Caso D |
| -------------------- | ------ | ------ | ------ | ------ |
| `O`                  | 10.496 | 21.826 | 25.688 | 50.179 |
| `FEATURE_SUPERFICIE` | 603    | 1.220  | 1.440  | 2.760  |
| `FEATURE_PREZZO`     | 583    | 1.178  | 1.422  | 2.772  |
| `FEATURE_TITOLO`     | 500    | 1.000  | 1.250  | 2.500  |
| `FEATURE_LOCALI`     | 500    | 999    | 1.248  | 2.494  |
| `FEATURE_ASCENSORE`  | 493    | 982    | 1.244  | 2.505  |
| `FEATURE_BAGNI`      | 477    | 950    | 1.194  | 2.409  |
| `FEATURE_PIANO`      | 441    | 810    | 1.076  | 2.096  |
| `FEATURE_CANTINA`    | 307    | 363    | 886    | 1.521  |
| `FEATURE_BALCONE`    | 265    | 541    | 841    | 1.574  |
| `FEATURE_TERRAZZO`   | 215    | 447    | 455    | 948    |
| `FEATURE_LUSSO`      | 179    | 285    | 313    | 510    |
| `FEATURE_ARREDATO`   | 176    | 214    | 362    | 692    |
| `FEATURE_DATA`       | —      | —      | —      | 3      |

### 🧠 Osservazioni sulla distribuzione
   - Il dataset cresce in dimensione da A → D.
   - Le proporzioni fra le classi restano abbastanza costanti, ma alcune feature rare (es. FEATURE_LUSSO, FEATURE_ARREDATO) aumentano solo linearmente, rimanendo comunque sottorappresentate.
   - La label O domina con oltre l’80% dei token → squilibrio di classe da gestire (es. con pesi o campionamento).

---
## 13. Azioni Possibili per la Prossima Iterazione
### 📌 Revisione della logica di labeling
   - Passaggio da token labeling isolato a segmenti/frasi complete → più semantica, meno rumore.
   - Eventuale pre-annotazione automatica seguita da correzione manuale.

### 📌 Bilanciamento delle Classi
   - Usare class_weight="balanced" o loss pesate per penalizzare l’errore sulle classi sottorappresentate.
   - Considerare tecniche di oversampling mirato su frasi rappresentative di FEATURE_*.

### 📌 Logging di errori per classe
   - Introdurre routine di logging degli errori per classe e pattern lessicali → utile per debug e data cleaning.






