# Log

# 31/05/2021

- Incontro con Davide per discutere la funzione
  [`apply_lstm_phrases`](https://github.com/lparolari/Loss_VT_Grounding/blob/0f5627dc9dda6edbe2be130b1eaf5a3fc98cc171/model_code/model.py#L117)
  e più nello specifico del funzionamento di `torch.pack_padded_sequences`,
  `torch.pad_packed_sequences` e `torch.gather`.

**Resources**

- [lstm_embeddings.py](model/lstm_embeddings.py)
- [torch_gather.py](model/torch_gather.py)
- [torch_sum.py](model/torch_sum.py)

# 28/05/2021

- Setup zsh su lab tesisti
- Setup dell'ambiente per lanciare il training su lab tesisti
- Mail a Luca Righi per aggiornamento versione di CUDA 10.1 su lab tesisti così
  da poter trainare il modello direttamente sul lab
- [Approfondimento](model/text_embeddings.py) sulla funzione degli embedding per
  il testo
  [create_embeddings_network](https://github.com/lparolari/Loss_VT_Grounding/blob/0f5627dc9dda6edbe2be130b1eaf5a3fc98cc171/model_code/model.py#L165)
- [Approfondimento su LSTM](model/lstm_embeddings.py)
- [Approfondimento su tokenizer](model/text_tokenizer.py)

**Resources**

- [Compiling and installing Zsh without root privileges on Stanford's Sherlock (https://sherlock.stanford.edu) for use in tmux](https://gist.github.com/mgbckr/b8dc6d7d228e25325b6dfaa1c4018e78)
- [text_embeddings.py](model/text_embeddings.py)
- [lstm_embeddings.py](model/lstm_embeddings.py)
- [text_tokenizer.py](model/text_tokenizer.py)

# 26/05/2021 - 27/05/2021

- Test del modello in locale (dati montati in locale)
- Setup dei symlink su cluster
- Debug di un file `.pickle`
- Generazione dei file preprocessed tramite job sul cluster
  `~/Jobs/vtg-make-dataset-cpu1.job`
- Setup dell'ambiente di lavoro parallelo su lab tesisti
- Prima esecuzione del modello (non completa, errore sul vocab)

**Resources**

- [.pickle viewer](resources/pickle_viewer/pickle_viewer.py)
- [setup_storage.sh](resources/setup_storage/setup_storage.sh)
- [mount.sh (locale)](resources/mount/mount.sh)
- [vtg-make-dataset-cpu1.job](resources/jobs/vtg-make-dataset-cpu1.job)

# 25/05/2021

- Studio di PyTorch
- Studio del codice di Davide
- Incontro con Davide per paralre del codice

# 24/05/2021

Studio di PyThorch e esperimenti su notebook colab seguendo i loro
[tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html).

# 22/05/2021

Aggiunta delle descrizioni ed esempi per i _raw data_ su flickr e referit, più
alcune info e riferimenti per i dataset _flickr_, _referit_ e l'ontologia
_YAGO_.

**Resources**

- [Dataset Cheatsheet](datasets/README.md)
- Generate a deliverable document from the dataset cheatsheet with
  `pandoc README.md -o README.pdf`.

# 21/05/2021

- Configurazione dell'ambiente sul cluster
- Studio del codice di preprocessing del dataset per capire gli input del
  modello

**Resources**

- [~~flickr30k_raw cheatsheet~~](datasets/flickr30k_raw.md)

# 20/05/2021

- Lettura paper self-supervised
- Lettura documento utilizzo cluster

**Resources**

- [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- [Cluster Usage](http://computing.math.unipd.it/highpc/clusterusage)

# 19/05/2021

Attivazione account per il cluster e lab tesisti.

# 18/05/2021

Riunione con Davide per discutere del suo paper e organizzarsi per il lavoro sul
codice.

**Todo**

- [ ] Mail ai prof per spazio di lavoro sul cluster
- [ ] Setup conda, Python e dipendenze codice paper
- [ ] Download datasets
- [ ] Setup PyCharm

**Resources**

- [A Better Loss for Visual Textual Grounding](https://drive.google.com/file/d/1_jHFoli3DpwJXNF0N4vr2AQpmmtOGzkY/view?usp=sharing)
- [KL divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)

# 17/05/2021

Lettura e studio del paper di Rigoni. Letturadi un nuovo paper
sull'apprendimento self-supervised.

# 13/05/2021 (\*)

> Milestone: presentazione dei paper ai prof

Esposizione dei due paper (Align2Ground e Without Pairs) ai prof. Difficoltà
nell'esposizione per via della puntualizzazione sull'aspetto tecnico dei prof.

# 11/05/2021

Domande sui paper a Davide in vista della presentazione con i prof e scelta dei
paper da presentare.

# 04/05/2021 - 05/05/2021

Presentazione dei sei paper a Rigoni. Bene in generale ma si è faticato sulla
comprensione del modello di alcuni di questi. Due paper dei sei sono stati
scartati per essere out-of-scope (Referring Expression Grounding e Linguistic
Strucutre)

# 27/04/2021 - 03/05/2021

- Lettura dei paper di riferimento per il dataset e per il task di VT-grounding
  supervised.
- Preparazione slides sui sei paper dell'ambito wekaly/un-supervised su
  VT-grounding e studio degli stessi.

# 26/04/2021

Incontro con Rigoni per la definizione del lavoro e nuovi paper sul tema scelto
da leggere (email principale).

# 22/04/2021 (\*)

> Milestone: scelta dell'argomento

Lettura paper VT-grounding weakly-supervised e scelta di questo ambito di
lavoro.

# 21/04/2021

Colloquio individuale con Rigoni per definire meglio gli obiettivi e il contesto
della tesi.

# 20/04/2021

Incontro con Ballan, Sperduti e Rigoni in cui si è discusso dei possibili
obiettivi di questa tesi. Tra le proposte (VQA, VT-grounding, VTKEL, software
gestione errori modello VT-grounding) si è parlato principalmente di
VT-groundin, tema oggetto di studi del dottorando che mi seguirà e più
abbordabile in termini di quanto lavoro c'è da fare per una tesi.

# 19/04/2021

Lettura paper e in vista del colloquio con Ballan, Sperduti e Rigoni.

# 30/04/2021 (\*)

> Milestone: inizio

Incontro conoscitivo con Ballan e discussione sugli argomenti di tesi.
