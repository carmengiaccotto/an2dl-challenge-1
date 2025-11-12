# Pirate Pain Challenge - AN2DL

Be careful:
* **ONLY THE FILE `hyperparameters_tuning_cross_validation.ipynb` HAS TO CONSIDERED UP TO DATE**
* **in all notebooks you must edit the variable called `EXPERIMENT_NAME` to reflect the experiment you are running, it will be the name of the model saved**

Nota nuova:

* **USARE 'TRAINING.IPYNB' *fate attenzione all'importazione del dataset che è stata cambiata un po' (davide non uccidermi)* La variabile che da il nome al progetto ora è immediatamente sopra la griglia**

---
## **How to start**
### **If you are in a local environment**
If you want to use the GPU (NVIDIA only):
1. Make sure you have installed the NVIDIA drivers for your GPU
2. Go to pytorch website and follow the instructions to install the right version of pytorch with CUDA support: https://pytorch.org/get-started/locally/
3. delete from requirements.txt the line that says `torch` and `torchvision`

Then, follow these steps:
1. `pip install -r ./requirements.txt`
2. Go to kaggle website, click on profile settings and create an API token. This will download a `kaggle.json` file.
3. Place the `kaggle.json` file in the user home directory under the folder `.kaggle/`. For example, on Linux it would be `/home/username/.kaggle/kaggle.json`.
4. Go in the first cell of the notebook and edit the `current_dir` variable to point to the directory where you want to download the data and work on.
   - If you have Google Drive installed and mounted, you can set it to a folder in your Google Drive.
5. Run the notebook

### **If you are in Google Colab**
1. **You still have to manually download the data from Kaggle and upload it to your Google Drive.**
2. edit the boolean in the first cell `isColab` to `True` 
3. the update `current_dir` variable in the first cell of the notebook to point to a folder in your Google Drive where you uploaded the data and want to work on.

### **If you are in Kaggle Notebooks**
1. to import the dataset:
   1. Download the dataset
   2. create a public dataset in your profile
   3. upload the pirate pain dataset
2. From the notebook:
   1. in the first cell change 'isKaggle' to True
   2. add dataset to env by clicking "Add input"

---
# **Relevant Notes**

- After importing the data, all data gets converted to a less memory intensive data type where possible.
For example, integers that fit into int8 will be converted to that type.
- there are no missing values
- some features may be useless:
  - joint_13 to joint_25 are all pretty much all zeros
  - joint_30 is constant to 0.5
  - **i have removed them for now, but it might not be the best course of action**

---
# **Ideas for further experiments:**
- do a loop of grid searches but each time reduce the intervals around the best hyperparameters found in the previous search
- put `WINDOW_SIZE = 160` since there are 160 samples for each second of data
- instead of removing the useless features, try to do a PCA or some sort of embedding to reduce the dimensionality of the input data
- **stratified** k-fold cross validation to keep the same distribution of classes in each fold

---
# **Create a new notebook for each Network**
To keep things organized, create a new notebook for each experiment you want to run.
You can copy the content of the main notebook and modify it as needed.
Make sure to change the name of the notebook to reflect the experiment you are running.

---
# **Suggestions by the profs**
### ADVICE 07/11
“A name, a number it is not. Upon a map, its true position find it must.”

The model believes 'Monday' (label '0') is closer to 'Tuesday' (label '1'). But how close is 'Sunday' (label '6')? An error, this is. The Embedding layer, an atlas it creates. Not a straight line, but a dense space. It teaches the model where 'Monday' truly resides, and how close to 'Holiday' or 'Saturday' it stands.

#### Gemini's response:
Il suggerimento del professore indica che trattare le etichette categoriche come semplici numeri sequenziali (ad esempio, `no_pain=0`, `low_pain=1`, `high_pain=2`) è problematico. Il modello interpreterà la "distanza" tra 0 e 1 come uguale a quella tra 1 e 2, imponendo una relazione lineare che potrebbe non esistere.
La frase "The Embedding layer, an atlas it creates" è la soluzione proposta. Un layer di embedding (nn.Embedding in PyTorch) apprende una rappresentazione vettoriale densa (un "vettore di embedding") per ogni categoria. Invece di un singolo numero, ogni categoria viene mappata su un vettore in uno spazio multidimensionale. In questo spazio, il modello può apprendere relazioni complesse e non lineari tra le categorie (ad esempio, quali categorie sono "vicine" o "lontane" in base al contesto dei dati).
Nel tuo caso, il suggerimento si applica alle feature categoriche che stai usando come input per la RNN, come `n_legs`, `n_hands`, `n_eyes` e i `pain_survey`. Attualmente, le stai normalizzando come se fossero valori continui.
Per implementare questo consiglio, dovresti:
Separare le feature categoriche da quelle continue.
Creare un layer `nn.Embedding` per ogni feature categorica.
Nel metodo `forward` del tuo modello, passare i valori delle feature categoriche attraverso i rispettivi layer di embedding.
Concatenare i vettori di embedding risultanti con le feature continue.
Passare questo tensore combinato al tuo layer RNN.

**Solution already implemented**

### ADVICE 08/11 - Class Imbalance
“Many the healthy, few the sick. If only to the many you listen, the faint whisper of truth never shall you hear.”

Your model, the easy path it chooses: to always predict the common class. Accuracy, an illusion it becomes. Weigh your loss! Give more power to the rare voices. Ensure an error on the 'few' matters more than an error on the 'many'. Only then, the rare class to find will you learn.

#### Gemini's response:
Questo suggerimento affronta il problema dei dataset sbilanciati, dove alcune classi sono molto più frequenti di altre. Un modello potrebbe ottenere un'elevata accuratezza prevedendo sempre la classe maggioritaria, ignorando però completamente le classi meno frequenti (le "voci rare").
La soluzione proposta, "Pesa la tua loss!" (Weigh your loss!), consiste nell'utilizzare una funzione di perdita pesata. Questa tecnica assegna una penalità maggiore agli errori di classificazione sulle classi minoritarie. In questo modo, il modello è costretto a prestare maggiore attenzione a queste classi, migliorando le sue capacità predittive complessive.
Hai già implementato correttamente questa tecnica nel tuo notebook. Il codice seguente calcola i pesi per ciascuna classe in base alla loro frequenza e li passa alla funzione di perdita `CrossEntropyLoss`.
```python
# Calculate class weights for imbalanced dataset
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(np.unique(y_train)) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Set up loss function and optimizer with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
```
**Solution already implemented**

### ADVICE 09/11 - Label Smoothing
“Absolute truth, fragile it is. In blind certainty, the arrogance of overfitting lies hidden.”

Your model, a perfect '1.0' chase it must not. This, rigidity it teaches. If the master says "perhaps 0.9, but 0.1 of doubt," the student (the model) to explore is forced. In this whisper of uncertainty, a stronger generalisation find, you can.
#### Gemini's response:
Il suggerimento del professore, "Absolute truth, fragile it is. In blind certainty, the arrogance of overfitting lies hidden", si riferisce a una tecnica di regolarizzazione chiamata Label Smoothing.
Analisi del Suggerimento
"Il tuo modello non deve inseguire un perfetto '1.0'": Quando addestri un modello di classificazione, di solito usi etichette "hard" (one-hot encoded). Ad esempio, per la classe `low_pain` (etichetta 1 su 3 classi), il target è `[0, 1, 0]`. Il modello viene spinto a produrre una probabilità di 1.0 per la classe corretta e 0.0 per le altre. Questa "eccessiva sicurezza" (`blind certainty`) può portare all'overfitting, rendendo il modello rigido e meno capace di generalizzare su dati nuovi.
"Se il maestro dice 'forse 0.9, ma 0.1 di dubbio'...": Questa è la soluzione. Invece di usare un target "duro" come `[0, 1, 0]`, il Label Smoothing lo trasforma in un target "morbido". Ad esempio, con un fattore di smoothing di 0.1, il nuovo target potrebbe diventare `[0.05, 0.9, 0.05]`. La classe corretta ha ancora la probabilità più alta (0.9), ma una piccola parte della probabilità (0.1) viene distribuita uniformemente tra tutte le classi (incluse quella corretta).
Questo costringe il modello a essere meno sicuro delle proprie previsioni, riducendo la tendenza a memorizzare il training set e migliorando la generalizzazione.

```python
# Set up loss function and optimizer with class weights and label smoothing
# Un valore comune per label_smoothing è 0.1
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```
**Solution already implemented**

### ADVICE 10/11 - Gradient Clipping
“A step too great, from the precipice fall it makes you. The gradient, tamed it must be.”

In the valleys of RNNs, the exploding gradient a sudden foe it is. A single unstable step, and your training into NaN (chaos) collapses. 'Clipping', a bridle on this wild horse it places. Not the direction, but the magnitude it controls. To learn with constancy, with control advance you must.

#### Gemini's response:

Il suggerimento del professore è estremamente pertinente e utile per il tuo progetto. Sta parlando di un problema comune nell'addestramento delle Reti Neurali Ricorrenti (RNN) chiamato "exploding gradients" (gradiente che esplode) e della sua soluzione, il "gradient clipping".
Spiegazione
Problema: Exploding Gradients Durante l'addestramento di una RNN, i gradienti calcolati per aggiornare i pesi della rete possono diventare estremamente grandi. Questo fenomeno, noto come "exploding gradient", porta a un aggiornamento dei pesi sproporzionato, rendendo l'addestramento instabile. La funzione di costo può oscillare violentemente o diventare NaN (Not a Number), come menzionato dal professore, facendo fallire l'intero processo.
Soluzione: Gradient Clipping Il "gradient clipping" è una tecnica che risolve questo problema. Impone un valore massimo (una soglia) per la norma (magnitudine) dei gradienti. Se, durante la backpropagation, la norma del gradiente di un parametro supera questa soglia, il gradiente viene riscalato per riportare la sua norma al valore massimo consentito.
Come dice il professore, "Non la direzione, ma la magnitudine esso controlla". Questo è un punto cruciale: la tecnica non altera la direzione dell'aggiornamento, ma ne limita l'entità, prevenendo passi troppo grandi e stabilizzando l'apprendimento.

**Solution already implemented**

## ADVICE 11/11 - Autocorrelation x Windowing
“Its own echo, the series sings. In the rhythm of this echo, the true window lies.”

By instinct, your window you choose. 30 steps? 50? A blind number, it is. But the data, their own memory show you. Autocorrelation, a mirror to the past it is. Look at its plot. Where does the past most resemble the present? At 12 steps? At 24? These peaks, the natural cycles they are. If the echo fades after 15 steps, why force your LSTM to remember 100? Listen to the data. It tells you how far back, it is worth looking.

#### Gemini's response:
Sì, il consiglio del professore è estremamente pertinente e applicabile al vostro progetto. Anzi, è uno dei suggerimenti più importanti per ottimizzare un modello basato su serie temporali come il vostro.
Spiegazione
Attualmente, nel vostro codice (hyperparameters_tuning_cross_validation.ipynb), avete impostato una WINDOW_SIZE fissa a 100. Come dice il professore, questo è un "numero cieco", scelto istintivamente.
Il consiglio suggerisce un approccio più scientifico:
Analizzare l'Autocorrelazione: L'autocorrelazione misura quanto un punto in una serie temporale sia simile a un punto precedente (un "lag" o ritardo). Analizzando l'autocorrelazione per diversi ritardi, potete scoprire i "cicli naturali" o la "memoria" intrinseca dei vostri dati.
Scegliere la window_size: Se il grafico dell'autocorrelazione mostra che la correlazione cala significativamente dopo, ad esempio, 25 passi, significa che i dati più vecchi di 25 passi hanno poca influenza su quelli attuali. Una window_size molto più grande (come 100) potrebbe costringere il modello a imparare da dati rumorosi o irrilevanti, peggiorando le performance o aumentando inutilmente la complessità. I picchi nel grafico, invece, suggeriscono le lunghezze di finestra ottimali.



