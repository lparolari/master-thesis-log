# Log

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [02/07/2021 - Test Loss e Masking](#02072021---test-loss-e-masking)
- [01/07/2021 - Implementazione loss](#01072021---implementazione-loss)
- [29/06/2021 - Call con Davide per problemi sulle loss](#29062021---call-con-davide-per-problemi-sulle-loss)
- [28/06/2021 - Risoluzione problemi referit / ottimizzazione loss](#28062021---risoluzione-problemi-referit--ottimizzazione-loss)
- [22/06/2021 - Inizio adattamento per referit](#22062021---inizio-adattamento-per-referit)
- [21/06/2021 - Call per aggiornamento con il prof.](#21062021---call-per-aggiornamento-con-il-prof)
- [18/06/2021 - Test Example Procedure](#18062021---test-example-procedure)
- [17/06/2021 - Learning Unsupervised Visual Grounding Through Semantic Self-Supervision](#17062021---learning-unsupervised-visual-grounding-through-semantic-self-supervision)
- [16/06/2021 - Accuracy e Pointing Game Accuracy](#16062021---accuracy-e-pointing-game-accuracy)
- [15/06/2021 - Loss: IoU e accuracy](#15062021---loss-iou-e-accuracy)
- [11/06/2021 - Black friday: problema di memoria / problema chunk vuoti](#11062021---black-friday-problema-di-memoria--problema-chunk-vuoti)
- [10/06/2021 - Input negativi e loss discriminativa](#10062021---input-negativi-e-loss-discriminativa)
- [09/06/2021 - Modifica funzioni di padding e loss, flickr30k pico, github action](#09062021---modifica-funzioni-di-padding-e-loss-flickr30k-pico-github-action)
- [08/06/2021 - Modifica dataloader e funzioni di padding](#08062021---modifica-dataloader-e-funzioni-di-padding)
- [07/06/2021 - Progettazione modello Weakly-Supervised](#07062021---progettazione-modello-weakly-supervised)
- [04/06/2021 (x) - Discussione sul lavoro da fare sul codice](#04062021-x---discussione-sul-lavoro-da-fare-sul-codice)
- [03/06/2021](#03062021)
- [01/06/2021](#01062021)
- [31/05/2021](#31052021)
- [28/05/2021](#28052021)
- [26/05/2021 - 27/05/2021](#26052021---27052021)
- [25/05/2021](#25052021)
- [24/05/2021](#24052021)
- [22/05/2021](#22052021)
- [21/05/2021](#21052021)
- [20/05/2021](#20052021)
- [19/05/2021](#19052021)
- [18/05/2021](#18052021)
- [17/05/2021](#17052021)
- [13/05/2021 (\*)](#13052021-%5C)
- [11/05/2021](#11052021)
- [04/05/2021 - 05/05/2021](#04052021---05052021)
- [27/04/2021 - 03/05/2021](#27042021---03052021)
- [26/04/2021](#26042021)
- [22/04/2021 (\*)](#22042021-%5C)
- [21/04/2021](#21042021)
- [20/04/2021](#20042021)
- [19/04/2021](#19042021)
- [30/04/2021 (\*)](#30042021-%5C)

</details>
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 02/07/2021 - Test Loss e Masking

# 01/07/2021 - Implementazione loss

- Alcuni fix nelle loss e calcolo logits

- Call con Davide per implementazione loss attrattiva-repulsiva presa dal paper
  [Clustering-Oriented Representation Learning with Attractive-Repulsive Loss](https://arxiv.org/pdf/1812.07627.pdf)
  con cosine similarity

- Inizio scrittura test funzioni loss

# 29/06/2021 - Call con Davide per problemi sulle loss

- I logits sono calcolati prima dell'embedding semantico, che viene utilizzato
  invece nella loss. Questo può dare problemi perché le rappresentazioni sono
  diverse e il modello fa fatica ad apprendere

- I pesi del layer linear che utilizzo come logits non sono mai modificati,
  i.e., il modello non apprende

- Selezionare la migliore bounding box non è corretto nel mio caso, non posso
  assumerla come ground truth perché altrimenti il modello viene ottimizzato per
  identificare come bb la prima che trova per via dell'inizializzazione causale
  dei pesi

- Possibile problema di overflow se si adottano le soluzioni ai problemi sopra
  riportati

# 28/06/2021 - Risoluzione problemi referit / ottimizzazione loss

- Troubleshoot dell'eccezione pickle "invalid key \x00"
- Troubleshoot dell'eccezione pickle "memory error"

Il problema è dato dalla generazione dei file pickle durante la fase make
dataset. La soluzione è stata analizzare tutti i file e rigenerare quelli
corrotti.

Primo training completo di referit iniziato e completato con successo.

**Resources**

- Semantic Instance Segmentation with a Discriminative Loss Function
  [[paper]](https://arxiv.org/pdf/1708.02551.pdf)
  [[code]](https://github.com/nyoki-mtl/pytorch-discriminative-loss)
- [A friendly introduction to Siamese Networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)

# 22/06/2021 - Inizio adattamento per referit

# 21/06/2021 - Call per aggiornamento con il prof.

# 18/06/2021 - Test Example Procedure

Aggiustato la modalità test exaample per i miei requisiti e review di alcune
parti di codice seguendo il paper (loss).

# 17/06/2021 - Learning Unsupervised Visual Grounding Through Semantic Self-Supervision

Create le slide per il paper "Learning Unsupervised Visual Grounding Through
Semantic Self-Supervision" [[arxiv](https://arxiv.org/pdf/1803.06506.pdf)],
studio del paper in questione e possibili problemi/domande.

# 16/06/2021 - Accuracy e Pointing Game Accuracy

- Aggiunto il calcolo dell'accuracy
- Aggiunto il calcolo della pointing game accuracy
- Trovata una soluzione al problema della memoria (i.e., lanciare il training
  con num_workers=1, prefetch_factor=1)
- Fix phrases_2_crd tutto zeri (problema di tipo)
- Lanciato il primo vero training su lab tesisti

# 15/06/2021 - Loss: IoU e accuracy

- Fixato il problema del padder discusso nella issue #7
- Altri test sull'utilizzo della memoria. E' emerso che davide con
  `num_workers=1` e `prefetch_factor=1` occupa al massimo 8GB di RAM, quindi il
  problema proviene da qualche modifica nel mio codice. Da vedere le funzioni di
  pytorch `detach` e `item`, dovrebbero servire a scollegare le strutture di
  torch dal grafo delle computazioni e liberare memoria
- Investigato il problema della loss negativa, non dovrebbe essere un errore
- Ripristinato il calcolo del gradiente sulla loss calcolata da me
- Aggiunto il codice per il calcolo dell'IoU e dell'accuracy (da terminare)
- Inizio prove con il dataset referit e script per il download dei dataset

# 11/06/2021 - Black friday: problema di memoria / problema chunk vuoti

- Debug del problema di memoria che fa crashare il trianing: si è scoperto che
  in totale a quanto pare servono più di 8 GB per il training, soprattutto se
  nello stesso batch si caricano sia esempio negativo che positivo
- Trovato un esempio con chunk vuoti, da investifare il motivo (vedere issue
  github)

# 10/06/2021 - Input negativi e loss discriminativa

- Modifica al modello per il supporto all'input negativo
- Embedding per il codice delle varie features di modo da poterle confrontare
- Impiego della loss discriminativa (paper Xiao et al.)
- Prova di esecuzione su lab tesisti, senza successo per via dell'utilizzo della
  ram sempre crescente
- Alcuni fix

# 09/06/2021 - Modifica funzioni di padding e loss, flickr30k pico, github action

- Correzione del problema di matching forzato tra i chunk e le frasi gt del
  modello: rigenerazione del nuovo dataset che non forza il matching tra i chunk
  e le frasi (questo processo deve essere rovesciato in fase di validazione del
  modello)
- Modifica make_dataset e generazione del nuovo dataset sul cluster
- Integrazione del modulo padder e inizio costruzione della funzione di loss
- Creazione di _flickr30k_pico_
  ([Google Drive](https://drive.google.com/file/d/1aMXIHGxoADGjSH0TvhIxXrCNpedEBdB3/view?usp=sharing),
  ~ 80MB), un sottoinsieme di 10 esempi per train, test e validation
- Creazione di una github action che esegue il modello su _flickr30k_pico_

**Resources**

- [flickr30k_pico (WIP)](resources/flickr30k_pico/go.sh)
- [text_tokenizer_modularized.py](model/text_tokenizer_modularized.py)
- [torch_argmax.py](model/torch_argmax.py)
- [torch_view.py](model/torch_view.py)

# 08/06/2021 - Modifica dataloader e funzioni di padding

- Modifica della funzione `collate_fn` e per il supporto ai chunk
  (positivi/negativi) di ewiser anzichè le query
- Introduzione del modulo `padder` con funzioni di utilità per "paddare" tutto
  (frasi, chunk positivi/negativi, entità yago) alla stessa lunghezza
- Collezione del dataset "flickr30k pico" composto da 10 esempi per train, test
  e validation

# 07/06/2021 - Progettazione modello Weakly-Supervised

- Plannig dei lavori da fare per il modello weakly-supervised
- Identificazione delle porzioni di codice da modificare

# 04/06/2021 (x) - Discussione sul lavoro da fare sul codice

> Milestone: incontro per discussione sul lavoro vero e proprio

- Review del codice nuovo del modello e setup ambiene per nuovo repository
- Incontro con Davide per nuovi obiettivi di lavoro (rendere il modello
  weakly-supervised), si veda resoconto che segue
- Email a Ballan con aggiornamenti ("Aggiornamenti sulla tesi", 04/06/2021
  16:46)

**Resoconto call con Davide**

Dall'incontro è emerso che la parte del KG per le immagini estrae una sola
entità (quella correlata alla classe) e per le query estrae una top-k di entità.
Per un discorso di efficienza le tutte le entità rilevanti (ovvero quelle
nominate nelle query e quelle delle classi delle proposal) sono allineate in
fase di preprocessing con le classi e ne viene specificata la relazione
(colegate, non collegate, relazione padre figlio...).

Le entità nel modello sono utilizzate solamente tramite gli embedding. Si prende
l'embedding della classe della proposal e top-k della query, si fa la
concatenezione e si usano assieme al resto del modello.

Per quanto riguarda rendere il modello weakly-supervised si è parlato invece
dell'utilizzo dei chunk di ewiser come query assieme alle proposal dell'object
detector per fare l'allenamento. Non avendo a disposizione la ground truth
diventa difficile da fare anche la valutazione. Per questo motivo la valutazione
va fatta collegando i chink alle query in qualche modo (per esempio lasciando
stare gli allineamenti di chunk che non hanno query), ovvero, l'algoritmo
inverso rispetto a quello che ha fatto Daivde.

In più, per quanto riguarda la loss, bisogna pensare a qualcosa che lavora sulla
ricostruzione e che sfrutta la repulsione delle feature nello spazio latente.
Per questo motivo bisognerà pensre ad un campionamento di esempi random
_negativi_ da includere in fase di training.

Riassunto problemi principali individuati: valutazione del modello e
campionamento esempi negativi.

# 03/06/2021

- Implementazione del calcolo della classification e regression loss

**Resources**

- [losses.py](model/losses.py)

# 01/06/2021

- Implementaizone dell'estrazione delle features per l'immagine
- Fix dello script `pickle_viewer`

**Resources**

- [image_features.py](model/image_features.py)

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
