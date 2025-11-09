# **How to start**
* in all notebooks you must edit in the last cell the variable `INFERENCE_RESULTS_CSV_NAME` with the new version number
* in all notebooks you must edit the variable called `experiment_name` to reflect the experiment you are running, it will be the name of the model saved

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
Currently, it does not work because there are some import issues, need to nest them inside an if

1. You don't need to download the data, it is already available in the Kaggle environment.
2. just run the notebook as is.
---
# **Relevant Notes**

- After importing the data, all data gets converted to a less memory intensive data type where possible.
For example, integers that fit into int8 will be converted to that type.
- there are no missing values
- some features may be useless:
  - joint_13 to joint_25 are all pretty much all zeros
  - joint_30 is constant to 0.5

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

### ADVICE 08/11
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

