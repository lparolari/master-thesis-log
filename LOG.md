# Log

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<details>
<summary>Table of Contents</summary>

- [16/07/2021 (x) - Stop ssh torre, fix `get_iou_score` loss](#16072021-x---stop-ssh-torre-fix-get_iou_score-loss)
- [14-15/07/2021 - Run dei modelli e8, e9, e10](#14-15072021---run-dei-modelli-e8-e9-e10)
- [13/07/2021 - Problemi di training](#13072021---problemi-di-training)
- [12/07/2021 - Modifica top-k loss e IndexError](#12072021---modifica-top-k-loss-e-indexerror)
- [09/07/2021 - Studio per utilizzo sentence nel modello](#09072021---studio-per-utilizzo-sentence-nel-modello)
- [08/07/2021 - Minor refactor and problem fixes](#08072021---minor-refactor-and-problem-fixes)
- [06/07/2021 - Similarity based attraction](#06072021---similarity-based-attraction)
- [05/07/2021 - Top-K repulsion loss](#05072021---top-k-repulsion-loss)
- [02/07/2021 - Test Loss e Masking](#02072021---test-loss-e-masking)
- [01/07/2021 - Implementazione loss](#01072021---implementazione-loss)
- [29/06/2021 - Call con Davide per problemi sulle loss](#29062021---call-con-davide-per-problemi-sulle-loss)
- [28/06/2021 - Risoluzione problemi referit / ottimizzazione loss](#28062021---risoluzione-problemi-referit--ottimizzazione-loss)
- [22/06/2021 - Inizio adattamento per referit](#22062021---inizio-adattamento-per-referit)
- [21/06/2021 (x) - Call per aggiornamento con il prof.](#21062021-x---call-per-aggiornamento-con-il-prof)
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

# 18/07/2021 - Problemi di maschere sulle loss?

- Abbiamo notato un possibile problema sul "mascheramento" dei valori calcolati
  dalla funzione `get_k_similar` in `losses.py`. La funzione `get_k_similar` con
  strategia _random_ è implementata come segue:

```py
def get_k_random(tensor: torch.Tensor, mask: torch.Tensor, *, k: int, dim: int = -1, device=None):
    """
    Returns `k` uniform random sampled values from `tensor` on `dim`, and `mask`. The index tensor is allocated on
    `device`.

    :return: A tuple of tensors (Tensor, LongTensor)
    """
    size = tensor.size()
    indices = torch.randint(size[-1], (*size[:-1], k), device=device)
    values = torch.gather(tensor, dim, indices)
    # noinspection PyTypeChecker
    values = torch.masked_fill(values, mask == 0, value=0)
    return values, mask
```

Si pensava ad un problema di mascheramento degli unici 3 (se k=3) valori
ritornati. In realtà non succede perchè **per ogni** chunk estraiamo k indici a
caso e poi la maschera è applicata ai chunk sintetici, quindi quelli non
sintetici non possono vedersi mascherate le 3 bboxes estratte.

- Aggiunta maschera per le bboxes gt

- Aggiunta maschera per le bboxes predette

# 16/07/2021 (x) - Stop ssh torre, fix `get_iou_score` loss

SSH torre down tutto il giorno.

Fix `get_iou_score` bug: vedere
[#43](https://github.com/lparolari/VTKEL-solver/issues/43) e
[#43](https://github.com/lparolari/VTKEL-solver/issues/42).

# 14-15/07/2021 - Run dei modelli e8, e9, e10

Vedere tabella esperimenti num. 8, 9 e 10.

# 13/07/2021 - Problemi di training

Il training non viene svolto in modo corretto (accuracy scende invece di
aumentare) e abbiamo deciso di investigare il problema.

(1) Avevo un bug nella collate*fn tale per cui mandavo avanti \_batch_size*
esempi positivi e 1 esempio negativo, il resto sui negativi diventava padding.
Questo problema è già stato risolto.

(2) Sugli score (sia positivi che negativi) il minimo è sempre 0, mentre il
massimo è corretto. Dopo aver calcolato la cosine similarity, l'unica cosa che
faccio prima della stampa è una masked fill per applicare la maschera.
Probabilmente devo controllare di aver fatto correttamente questa operazione,
altrimenti non mi spiego quello 0 come minimo.

(3) Non so se è dovuta al punto (2) ma osservando le statistiche per diversi
batch ho notato che il modello tente a portare le due medie a valori vicini allo
0, e direi che è il caso peggiore per il modello perché non impara.

(4) La media degli score positivi e negativi è spesso molto vicina (scarto
solitamente inferiore a 0.02), soprattutto dopo qualche batch. Mi viene da
pensare che le immaigni abbiano più influenza dei chunk positivi e negativi nel
calcolo della similarità.

UPDATE 1

Verso gli ultimi batch della prima epoca ho il min che a volte scende sotto lo
0, però di pochissimo: -0.01, -0.006, e così via. Ho anche notato però che il
max sia dei positivi che negativi è spesso vicino a 1 (>0.98).

UPDATE 2

Per il punto (2), la masked fill non influenza il min. Sicuramente maschera gran
parte del tensore, ma min=0 significa comunque che gli score calcolati erano
almeno 0, probabilmente maggiori di 0.

Per i punti (3) e (4) invece, dando un'occhiata al primo batch dell'epoca 2
abbiamo che, per gli score positivi nessuno ha valori molto negativi (< -0.5),
non sono pochi invece quelli con valori molto positivi (> 0.5). Considerando
solo il primo chunk (il batch era paddato a 5 chunk, ma ho preso in
considerazione solo il primo per non fare confusione) su un batch di 128 esempi
ne ho contati almeno 60 tali per cui la score è maggiore di 0.5 per 90 o più
bounding box.

Sugli esempi negativi, nessuna score ha valori molto negativi, mentre circa 30
esempi hanno una score superiore a 0.5 per 90 o più bounding box, sempre tenendo
conto solo del primo chunk.

Secondo me questa cosa implica un bias positivo sulla similarità che calcolo, in
qualche modo... La similarità viene calcolata sull'embedding dei chunk e delle
immagini effettuato tramite due torch.nn.Linear separati. C'è da contare però
che il layer delle immagini fa una riduzione effettiva delle features (da 2053 a
500), quello del testo invece no (da 500 a 500).

UPDATE 3

Controllare leaky relu \
Controllare feature bb con anche numeri negativi

UPDATE 4 - Possibili miglioramenti dopo conclusione del training

(1) se la parte delle feature delle bounding box è 2053 e sono tutte positive,
serve una rete più potente per dare più espressività e in caso permettere anche
valori negativi nelle trasformazioni successive di quelle features. Nel codice
l'unica trasformazione è data dalla rete che proietta le features a
dimensione 500. Qui c'è anche il problema che le features vengono mappate ad una
dimensione molto più piccola, perdendo molto probabilmente informazione

Questi due problemi assieme mi portano a pensare che forse è il caso di
ingrandire la potenza della rete. Questo si può fare aggiungendo layers non
Lineari con leaky relu oppure anche ingrandendo la dimensione di output.

Il testo invece ha una dimensione di 500 in output dopo la lstm. Questa è la
stessa che c'è nel layer che fa la proiezione. Dunque qui non credo ci siano
problemi di perdita di informazione

(2) poi ci sarebbe una parte minor su cui si potrebbe concentrarsi. Cioè le
features delle bounding box hanno concatenato anche le features spaziali (5
dimensioni). Queste sono molto importanti di solito in quanto dicono dove si
trova la bounding box nello spazio dell'immagine

Prova una cosa. Lascia andare il training con la versione random e vedi che
risultato dà. Vediamo se ci sono altri errori e in caso vediamo che fare.
Altrimenti è come andare alla ceca

Inoltre, nota che la LSTM è veramente molto espressiva e probabilmente se togli
l'ultimo layer di proiezione e usi direttamente l'output della rete, va meglio

Per cui metteresti la proiezione solo per le bounding box. Chiamo proiezione la
funzione dell'ultimo layer embed_phrases

Ma comunque prima dobbiamo vedere fino ad ora come si comporta. Poi in caso fai
delle modifiche, una alla volta, e vedi come si comporta. So che purtroppo è un
problema il tempo di training e probabilmente ti viene da fare più modifiche al
colpo, ma in pratica è meglio evitare

UPDATE 5

Certo, sono d'accordo e cercherò di spostare quantomeno le modifiche su branch
separate così da non fare confusione. In caso le modifiche le provo a tempo
perso sui pc del laboratorio.

Per il punto (1) il paper
[Weakly-supervised Visual Grounding of Phrases with Linguistic Features](https://arxiv.org/pdf/1705.01371.pdf)
(quello da cui abbiamo preso la prima versione della loss) ha la dim delle feat
testuali (LSTM) = 512, mentre per le feat visive applica un 2-layer perceptron e
poi va a matchare la dimensione delle altre tramite batch normalization. Quindi
tecnicamente ha una rete più potente di quella che ho implementato, anche se non
spiega come è fatta internamente.

Per il punto (2) invece suggeriresti di pesare maggiormente le ultime 5
features?

Per l'embedding dei chunk invece credo di aver assunto tempo fa che "non poteva
che migliorare" perché la rete avrebbe potuto apprendere. Invece anche nel paper
effettivamente usano direttamente le feat dell'LSTM.

# 12/07/2021 - Modifica top-k loss e IndexError

Call con davide per training che scende in termini di accuracy: modificata la
strategia top-k per spostare le feature degli esempi più simili in favore di una
strategia k-random.

Investigato il problema IndexError ottenuto sul padding dei concetti
all'epoca 4. Non si riesce a capire da cosa sia dato.

# 09/07/2021 - Studio per utilizzo sentence nel modello

# 08/07/2021 - Minor refactor and problem fixes

Lanciato di nuovo il training del modello con 3 pulsioni e attrazioni, era
crashato per un index out of bound.

# 06/07/2021 - Similarity based attraction

Calcolo della similarità tra il concetto di una frase e la classe predetta della
bounding box per poter effettuare l'attrazione delle features scalata con questa
similarità. Si veda [\#30](https://github.com/lparolari/VTKEL-solver/issues/30)
su GitHub.

# 05/07/2021 - Top-K repulsion loss

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

# 21/06/2021 (x) - Call per aggiornamento con il prof.

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
