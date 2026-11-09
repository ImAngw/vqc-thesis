# Report tecnico: testa VQC per una policy di RL

2026-10-15 · @Angelo

## 1. Sommario

Con pochi layout di training (K=25) la testa VQC J generalizza meglio di un MLP della stessa taglia anche a parità di entropia (+2.13 di ritorno, CI95 [0.82, 3.44]; risolve il 92% dei layout nuovi contro il 77%), e resta robusta al rumore dell'hardware IBM attuale (perdita 0.10 su Torino, 0.38 su Brisbane).

**Domanda.** Una policy di RL con testa quantistica variazionale (VQC, readout di Born: l'azione è l'esito di una misura) può superare una rete classica in un regime preciso, per una ragione identificabile? Su layout infiniti il VQC non raggiunge l'MLP grande (sezione 4). Il regime scelto è la **generalizzazione**: training su un insieme fisso di K layout del mondo, test su 100 layout mai visti.

**Risultati principali** (10 seed per cella, ritorno di test, differenze appaiate per seed con CI al 95%):

- **K=25, stessa taglia, stessa entropia:** J (117 parametri) batte l'MLP h5 (113 parametri) portato all'entropia di J: 4.05 contro 1.92, differenza +2.13 [0.82, 3.44]. È il risultato che resiste a tutti i controlli.
- **Curva in K:** il vantaggio di J sull'MLP della stessa taglia è netto a K=5 e K=25 e non si rileva a K=200 e K=inf. Con molti layout l'MLP grande (h128, 2696 parametri) vince: −0.77 e −1.16 per J.
- **K=5:** tutte le teste vanno in overfitting (ritorno di test negativo). Il vantaggio di J lì viene in gran parte dalla sua stocasticità: a pari entropia in valutazione scende da +3.97 a circa +1 e non è più rilevabile.
- **Perché J vince****:** una testa classica con feature di Fourier (120 parametri) eguaglia J nel ritorno e si comporta come J quanto due seed di J tra loro. Il VQC è una realizzazione di questo bias induttivo; non emerge un vantaggio specifico del circuito.
- **Rumore:** con le calibrazioni reali di IBM Torino e Brisbane, simulando ogni azione con Aer (1 shot per passo), J perde 0.10 (nessuna differenza rilevata) e 0.38 (9%).

**Limiti principali:** il rumore è applicato in esecuzione su policy addestrate senza rumore; nessun run su hardware reale.

## 2. Ambiente

L'agente deve raggiungere una cella obiettivo in una griglia 15×15 con ostacoli, muovendosi come il re degli scacchi e vedendo solo una finestra locale: è un problema parzialmente osservabile (POMDP). L'ambiente è un env Gymnasium a singolo agente scritto per questo progetto (`rl_env/env.py`, specifica in `rl_env/ENV_SPEC.md`, versione 13).

**Configurazione usata in tutti gli esperimenti di generalizzazione**: finestra di percezione 5×5, 30 ostacoli, al massimo 300 passi per episodio.

### Mondo e dinamica

- **Layout:** a ogni reset si estraggono posizione di partenza, obiettivo (a distanza di Chebyshev ≥ 8 dalla partenza) e 30 ostacoli distinti. Il layout resta fisso nell'episodio.
- **Azioni:** 8 direzioni del re (N, NE, E, SE, S, SW, W, NW). Se la destinazione è un ostacolo o è fuori dalla griglia l'agente resta fermo, e il passo conta comunque.
- **Fine episodio:** terminazione quando l'agente entra nell'obiettivo; troncamento dopo 300 passi.
- **Pool di layout (K):** con K > 0 l'env estrae una volta sola K layout con un generatore dedicato (seme del pool) e a ogni reset ne sceglie uno a caso. Il training usa il pool; valutazione e test usano sempre layout nuovi.

### Osservazione

Tabella 1. Tre componenti, nessuna delle quali rivela il layout completo:

| componente | forma | contenuto |
| --- | --- | --- |
| perception | 1×5×5 | finestra centrata sull'agente: libera 0, ostacolo o fuori griglia 1 |
| navigator | 17 valori | frazione di tempo trascorsa; posizione dell'agente; posizione relativa dell'obiettivo una volta scoperto (−1, −1 prima); per ognuno dei 4 settori N, E, S, W la frazione di celle inesplorate e la cella inesplorata più vicina |
| memory | 10×10 | le ultime 10 mosse: azione one-hot (8) e spostamento netto rispetto a 10 mosse prima (2) |

L'obiettivo è "scoperto" quando entra nella finestra di percezione. La memoria è l'osservabile di storia delle mosse.

### Reward

La reward per passo somma un termine di esplorazione, una penalità per passo e il premio di arrivo, con i valori della tesi precedente (premio obiettivo GRW = 5):

**r(t) = 0.5 · (GRW / N²) · n_new(t) − 1.5 · (GRW / T_max) + GRW · [obiettivo raggiunto al passo t]**

con N = 15, T_max = 300 e n_new le celle che entrano per la prima volta nella finestra. In pratica: +0.011 per cella nuova esplorata, −0.025 per passo, +5 all'arrivo. Un episodio che scade senza esplorare vale −7.5; un buon episodio vale circa 5. Gli altri termini previsti dall'env (penalità per urto, shaping sulla distanza dall'obiettivo) sono spenti.

### Metriche

- **Ritorno di test:** ritorno medio su 100 layout nuovi (semi 10000–10099, gli stessi a ogni valutazione e disgiunti da ogni pool di training), con azioni campionate dalla policy.
- **Tasso di successo:** frazione di quei 100 episodi che raggiungono l'obiettivo entro 300 passi.
- **Ritorno sul pool:** la stessa valutazione sui K layout di training.
- **Gap** = pool − test: misura di overfitting.
- **Due modi di valutare un run.** Media dei round 41–50: media delle valutazioni di fine round negli ultimi 10 round; è il numero delle sezioni 4–7 salvo indicazione. Pesi finali: una valutazione dei pesi del round 50 sugli stessi 100 layout; è la base delle analisi offline (controllo di temperatura in sezione 7, sezioni 8 e 9). I due numeri differiscono di qualche decimo: J a K=25 vale 4.05 nel primo modo e 4.18 nel secondo.

## 3. Algoritmo di RL e architettura

Tutte le teste sono addestrate con lo stesso A2C, lo stesso encoder e lo stesso budget di 7500 aggiornamenti: cambia solo la testa che produce le 8 probabilità d'azione.

[embedded content: Figura 1 · architettura della policy · encoder e critic condivisi, testa intercambiabile]

In Figura 1 l'unica parte che cambia tra gli esperimenti è la testa evidenziata; il critic legge le feature ma il suo gradiente non raggiunge l'encoder.

### A2C

Advantage Actor-Critic sincrono, la variante sincrona di A3C, implementato in `rl_train/`.

- **Rollout:** 16 ambienti in parallelo × 64 passi = 1024 transizioni per aggiornamento; azioni campionate dalla policy.
- **Ritorni:** n-step bootstrappati con il valore V, azzerati alla terminazione e bootstrappati con V(osservazione finale) al troncamento; vantaggio standardizzato per rollout.
- **Loss:** policy loss + 0.5 · value loss (MSE) − c_ent · entropia; c_ent decresce linearmente da 0.02 a 0.001 nei primi 2500 aggiornamenti, poi costante.
- **Ottimizzatore:** RMSprop, lr 7·10⁻⁴ (testa J: 2·10⁻³), clipping della norma del gradiente a 1.0, γ = 0.99. Il lr della testa VQC viene da una scansione nella fase di sviluppo (7·10⁻⁴, 2·10⁻³ e 7·10⁻³ sulla variante I; 3·10⁻³, 4·10⁻³ e 7·10⁻³ su J con il critic staccato): 7·10⁻³ è instabile, 3·10⁻³ e 4·10⁻³ non si distinguono da 2·10⁻³ con 5 seed. Fourier con lo stesso lr peggiora (2.76 ± 0.64 contro 3.68 ± 0.34 a K=25, 10 seed, 5 fallimenti su 10), quindi le teste classiche restano al lr di default; h5 e h128 non sono stati tarati.
- **Budget:** 7500 aggiornamenti in 50 round da 150. A fine round si applica un passo esterno di tipo Lookahead con momento di Nesterov (lr esterno 0.7, momento 0.9), come in [DiLoCo](https://arxiv.org/abs/2311.08105v3) con un solo worker: ogni run gira su una sola macchina (i run originali sono stati lanciati da uno strumento di orchestrazione non incluso nel repository). Lo pseudo-gradiente è arrotondato a bf16, e `train.py` riproduce lo schema con le opzioni `--outer-lr`, `--outer-momentum`, `--outer-dtype` e `--inner-steps`. Il passo esterno è uguale per tutte le teste: è un effetto del protocollo di training, non una scelta di metodo.
- **Valutazione:** a fine di ogni round, 100 episodi su layout nuovi e 100 sui layout del pool, in modalità campionata e greedy. Salvo indicazione, il numero riportato è la media dei round 41–50 (sezione 2, Metriche).

### Encoder condiviso e critic

L'encoder (410 parametri) trasforma l'osservazione in 12 feature in [−1, 1], 4 per ramo, ognuno chiuso da RMSNorm e tanh:

- **percezione M:** due convoluzioni 3×3 (1→2→2 canali, la seconda con stride 2), poi Linear → 4 (138 parametri);
- **navigator N:** Linear 17 → 4 (76 parametri);
- **memoria G:** GRU 10 → 4 sulle ultime 10 mosse, ultimo stato nascosto (192 parametri).

Il critic è Linear(12 → 64) – ReLU – Linear(64 → 1) (897 parametri). Due scelte valgono per tutte le teste:

- **critic staccato** (`detach_critic=1`): il critic legge le feature senza propagare il gradiente nell'encoder, che impara solo dalla policy. È la correzione che ha stabilizzato il VQC (sezione 4) e migliora anche l'MLP.
- **inizializzazione separata** (`split_init=1`): encoder e critic sono inizializzati da un flusso di numeri casuali separato da quello della testa, quindi a parità di seed tutte le teste partono dallo stesso encoder e dallo stesso critic.

### Le teste confrontate

Tabella 2. Le quattro teste confrontate. J è il nome della variante di circuito nel piano di lavoro, dove le varianti provate si chiamano I, J e K (sezione 4).

| testa | nome nel codice | parametri | struttura |
| --- | --- | --- | --- |
| J (VQC) | `vqc-nav-id` | 117 | 3 qubit, 3 caricamenti dei dati, readout di Born con codice Gray |
| Fourier | `fourier4` | 120 | 4 frequenze apprese W (4×12) → [sin Wx, cos Wx] → Linear(8 → 8) |
| MLP h5 | `mlp5` | 113 | Linear(12 → 5) – ReLU – Linear(5 → 8), stessa taglia di J |
| MLP h128 | `mlp128` | 2696 | Linear(12 → 128) – ReLU – Linear(128 → 8) |

**Il circuito J**** (Figura 2)****.** Tre blocchi r = 1, 2, 3 su 3 qubit. Nel blocco r il qubit i ha uno di tre ruoli, che ruota tra i blocchi: solo navigator, navigator + percezione, navigator + memoria. Ogni qubit riceve quattro porte di dati RY RZ RY RZ con angoli

**a(k) = w(r, i, k) · N(j) + c(r, ****ρ, k) · F****(j),  con j = (k + i) mod 4 e ****F = 0, M oppure G a seconda del ruolo ρ**** del qubit**

seguite da rotazioni addestrabili RZ RY RZ, una catena di CNOT (2,1), (2,0), (1,0), altre rotazioni addestrabili e una seconda catena; alla fine RY su ogni qubit. I 117 parametri: per blocco 12 pesi w, 8 pesi c (4 per la percezione e 4 per la memoria) e 18 angoli di rotazione addestrabili, cioè 3 × 38 = 114, più le 3 RY finali. Si misurano i 3 qubit: gli 8 esiti sono le 8 azioni (mappate con codice Gray, così due direzioni adiacenti differiscono di un solo bit) e la policy è la distribuzione di Born, logit = log P(esito). Agire richiede quindi **una sola misura per passo**. Il simulatore è in PyTorch puro (statevector, più un percorso a matrice densità per il rumore) ed è verificato contro un riferimento Qiskit scritto dalla descrizione del circuito (31 test, tra cui equivalenza numerica in float32 e float64 e gradienti).

[embedded content: Figura 2 · circuito J (vqc-nav-id) · un blocco con i ruoli del blocco 1, da quantum_heads/circuit_table_j.md]

Il navigator N entra su ogni qubit in ognuno dei tre blocchi; percezione e memoria entrano su un solo qubit per blocco, sommate al navigator nello stesso angolo.

**Perché Fourier.** Un circuito con caricamenti ripetuti dei dati è una serie di Fourier troncata negli angoli d'ingresso e può essere approssimato classicamente con feature di Fourier. La testa Fourier è il controllo classico di questa struttura, a parametri appaiati.

**La testa Fourier (****`fourier4`****).** Dalle 12 feature x dell'encoder calcola:

1. z = W x, con W ∈ ℝ⁴ˣ¹² addestrabile: 4 numeri, uno per riga. Ogni riga W_k è un vettore di frequenza, cioè la direzione e la frequenza di un'onda piana nello spazio delle feature.
2. [sin z, cos z]: 8 valori, seno e coseno di ciascuno dei 4 z_k.
3. Linear(8 → 8): gli 8 logit delle azioni, poi softmax.

Seno e coseno della stessa fase, combinati linearmente, danno un'onda di ampiezza e sfasamento qualsiasi (a·sin z + b·cos z = A·sin(z + φ)). La testa è quindi una combinazione lineare di 4 onde di cui impara direzione e frequenza (inizializzazione W ~ N(0, 1)), con 48 + 72 = 120 parametri. Rispetto a J ha meno frequenze ma libere, e la non linearità è nel softmax dei logit invece che nella probabilità di Born.

## 4. Sviluppo della testa VQC

Su layout infiniti il VQC, una volta stabilizzato, arriva a 4.02 di ritorno contro 5.11 dell'MLP h128 (10 seed): la capacità del circuito basta, il collo di bottiglia era la dinamica del training. Questa fase ha prodotto il circuito J e il setup usato poi nello studio di generalizzazione.

Tabella 3. Le fasi dello sviluppo della testa VQC, tutte a K=∞.

| fase | cosa è stato fatto | risultato |
| --- | --- | --- |
| 1. Progettazione del VQC | Readout di Born: l'azione è l'esito della misura dei 3 qubit, 1 shot per passo. Due varianti a 3 qubit con 3 caricamenti dei dati: I (`vqc-optim`, 93 parametri), con pesi moltiplicativi sulle feature; J (`vqc-nav-id`, 117 parametri), con il navigator su ogni qubit di ogni blocco. Simulatori PyTorch verificati contro Qiskit | In RL I arriva a 3.29 su 5 seed, lento e instabile, con policy piatta (entropia 1.6–1.7 contro 1.1–1.2 di h128); J inizialmente fa peggio (1.77) |
| 2. Distillazione per studiarne l'espressività | Studenti I e J addestrati a imitare (KL) la policy di un MLP h128, con encoder congelato o adattato insieme alla testa; 3 inizializzazioni | Con encoder adattato J raggiunge 4.97 di ritorno (96% del maestro, 5.18) e I 4.76; con encoder congelato J 4.53 e I 3.01. La capacità del circuito basta: il limite è la dinamica dell'RL |
| 3. Diagnosi causale | Warm start dagli studenti distillati; test su bonus di entropia, warm-up del critic, gradiente del critic | Il crollo delle policy VQC è causato dal gradiente del critic nell'encoder condiviso (5–8 volte quello della policy), che deforma gli angoli del circuito |
| 4. Critic staccato | `detach_critic=1` per tutte le teste, 10 seed | J 4.02 ± 0.28, h128 5.11 ± 0.15: J recupera, ma l'MLP grande resta davanti di 1.09 |

**Due conclusioni guidano il resto del lavoro.** La prima: il circuito J ha capacità sufficiente (in distillazione arriva al 96% del maestro), quindi il collo di bottiglia era il training, risolto in gran parte staccando il critic. La seconda: su layout infiniti non c'è parità con l'MLP grande, e ogni correzione aiutava anche l'MLP. Da qui la scelta di cercare un regime in cui la capacità limitata del VQC sia un vantaggio: pochi layout di training.

## 5. Regime di generalizzazione e metodo

Ogni testa è addestrata su un pool di K layout fissi e valutata su 100 layout mai visti, con K ∈ {5, 25, 200, ∞} e 10 seed per cella; il confronto tra teste è sempre appaiato per seed.

### Disegno

- **Pool condiviso:** il seed fissa sia l'inizializzazione dei pesi sia il seme del pool (`env_seed = layout_pool_seed = seed`). A parità di seed tutte le teste vedono gli stessi K layout e partono dallo stesso encoder.
- **K = ∞:** ogni reset estrae un layout nuovo (nessun pool). Per J si riusano i 10 run del setup con critic staccato della fase 4 (Tabella 3), dopo un controllo di riproducibilità (sezione 7).
- **Bracci:** J, Fourier, h5, h128 in tutte e quattro le K; a K=25 anche h5 e h128 a entropia appaiata (h5-ent, h128-ent): un coefficiente di entropia adattivo mantiene l'entropia di training a quella di J (1.425 nats; con 8 azioni il massimo è ln 8 ≈ 2.08).

### Statistica riportata

Si segue la pratica dei lavori di RL ([Henderson et al. 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694); [Colas et al. 2018](https://arxiv.org/pdf/1806.08295); [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264v4)): risultati su più seed come media con intervallo di confidenza, confronti letti dall'intervallo.

- **Per cella:** media ± errore standard (SE) sui 10 seed del ritorno medio nei round 41–50 (Tabella 4).
- **Confronto J − X:** differenza per seed (stesso pool), con il suo CI al 95% = media ± t₀.₉₇₅(n−1) · SE. L'appaiamento toglie dalla differenza la difficoltà del pool, che pesa su tutte le teste allo stesso modo.
- **Lettura:** CI che esclude 0 → "J meglio" o "X meglio"; CI che include 0 → "nessuna differenza rilevata", sempre con l'ampiezza del CI. Un CI che include 0 non viene letto come equivalenza.

## 6. Risultati: generalizzazione in funzione di K

Il vantaggio di J sull'MLP della stessa taglia esiste solo con pochi layout (K=5 e K=25) e sparisce da K=200; con molti layout l'MLP grande è la testa migliore (Figura 3, Tabella 4).

[embedded content: Figura 3 · ritorno di test in funzione di K · logs dei run, media dei round 41–50 · 10 seed per cella · linee tratteggiate: Fourier (lunghi), h5 (corti)]

Con 5 layout tutte le teste memorizzano e falliscono su layout nuovi; con 25 J e Fourier generalizzano, gli MLP no; da 200 in su tutte generalizzano e l'MLP grande è la migliore.

### Ritorni assoluti

Tabella 4. Ritorno di test (media ± SE su 10 seed, media dei round 41–50) e, tra parentesi, tasso di successo sui 100 layout nuovi.

| testa | K=5 | K=25 | K=200 | K=∞ |
| --- | --- | --- | --- | --- |
| J | −1.13 ± 0.37 (47%) | 4.05 ± 0.47 (92%) | 4.26 ± 0.30 (95%) | 4.02 ± 0.28 (94%) |
| Fourier | −5.10 ± 0.25 (13%) | 3.68 ± 0.34 (89%) | 4.33 ± 0.23 (95%) | 3.97 ± 0.30 (93%) |
| h5 | −3.79 ± 0.50 (24%) | 2.27 ± 0.28 (79%) | 3.73 ± 0.30 (93%) | 3.42 ± 0.53 (90%) |
| h128 | −5.33 ± 0.40 (11%) | 1.25 ± 0.46 (65%) | 5.02 ± 0.15 (97%) | 5.18 ± 0.13 (99%) |
| h5-ent |  | 1.92 ± 0.38 (77%) |  |  |
| h128-ent |  | 4.29 ± 0.23 (93%) |  |  |

Il ritorno segue soprattutto il tasso di successo: un episodio che scade vale circa −7.5, un arrivo circa +5. A K=25 J risolve il 92% dei layout nuovi, h5-ent il 77%.

### Differenze appaiate J − X

Tabella 5. Media della differenza per seed e CI al 95% (10 coppie per cella). In grassetto i CI che escludono lo 0.

| X | K=5 | K=25 | K=200 | K=∞ |
| --- | --- | --- | --- | --- |
| h5 (stessa taglia) | **+2.66 [1.82, 3.50]** | **+1.78 [0.58, 2.97]** | +0.53 [−0.52, 1.58] | +0.60 [−0.78, 1.98] |
| h5-ent (stessa taglia e entropia) |  | **+2.13 [0.82, 3.44]** |  |  |
| h128 | **+4.20 [3.44, 4.96]** | **+2.79 [1.40, 4.18]** | **−0.77 [−1.39, −0.14]** | **−1.16 [−1.81, −0.51]** |
| h128-ent |  | −0.24 [−1.17, 0.70] |  |  |
| Fourier | **+3.97 [3.21, 4.73]** | +0.37 [−0.82, 1.56] | −0.07 [−0.79, 0.64] | +0.05 [−1.13, 1.23] |

### Overfitting

Tabella 6. Gap = ritorno sul pool − ritorno di test, media ± SE (a K=∞ non c'è pool).

| testa | K=5 | K=25 | K=200 |
| --- | --- | --- | --- |
| J | 6.63 ± 0.39 | 0.84 ± 0.30 | 0.02 ± 0.04 |
| Fourier | 10.68 ± 0.24 | 1.13 ± 0.15 | 0.03 ± 0.05 |
| h5 | 8.81 ± 0.54 | 1.94 ± 0.27 | −0.04 ± 0.05 |
| h128 | 10.66 ± 0.42 | 4.35 ± 0.47 | 0.10 ± 0.03 |
| h5-ent |  | 1.60 ± 0.17 |  |
| h128-ent |  | 0.94 ± 0.18 |  |

### Lettura

- **K=25, la cella centrale.** J batte l'MLP della stessa taglia anche quando questo ha la stessa entropia (h5-ent, +2.13): il vantaggio non si spiega con la sola stocasticità della policy. Contro l'MLP grande a pari entropia (h128-ent) non si rileva differenza: J generalizza come un MLP 23 volte più grande regolarizzato all'entropia giusta.
- **Il regime conta.** Il CI di J − h5 sta sopra 0 a K=5 e K=25 e include 0 a K=∞: il vantaggio dipende dal numero di layout. Da K=200 h128 è davanti a J con CI sotto 0.
- **K=5: nessuna testa generalizza.** Tutte hanno ritorno di test negativo e un gap di 6–11: memorizzano i 5 layout. J perde meno degli altri, ma la sezione 7 mostra che questo dipende soprattutto dalla sua entropia. Il gap di J è più piccolo di quello di Fourier (6.63 contro 10.68, differenza appaiata −4.05, CI [−4.85, −3.26]), ma è lo stesso fatto della tabella precedente: sul pool le due teste sono pari (5.49 e 5.58), quindi la differenza di gap è quasi tutta la differenza di ritorno di test (+3.97) con il segno cambiato.
- **J e Fourier.** Nessuna differenza rilevata a K=25, 200 e ∞; a K=200 il CI è stretto ([−0.79, 0.64]), a K=25 e ∞ è largo e non esclude una differenza fino a circa 1.5 in un verso o nell'altro.
- **Variabilità tra seed.** J a K=25 ha due seed deboli (0.57 e 2.33 su 10); i fallimenti (ritorno < 3) sono 2 su 10 per J e 3 su 10 per Fourier.

## 7. Controlli

L'entropia della policy spiega quasi tutto il vantaggio di J a K=5 e sul grande MLP a K=25, ma non quello sull'MLP della stessa taglia a K=25.

### Entropia come confondente

Il ritorno di test è misurato campionando le azioni. Le MLP tendono a collassare su policy quasi deterministiche (entropia finale a K=5: h128 0.01, Fourier 0.04, h5 0.34), mentre J resta stocastico (0.67). In un POMDP la migliore policy stazionaria può essere stocastica, quindi più entropia può dare più ritorno su layout nuovi a prescindere dalla testa.

- **A K=25, durante il training:** h5 e h128 riaddestrati con l'entropia tenuta a quella di J. h128-ent sale da 1.25 a 4.29 e raggiunge J; h5-ent resta a 1.92, e J lo batte di +2.13 (Tabella 5).

### Ritorno greedy

Con l'azione più probabile (argmax) invece del campionamento quasi tutte le teste falliscono sui layout nuovi. Media dei round 41–50, 10 seed:

Tabella 8. Ritorno di test con azione greedy (argmax).

| testa | K=5 | K=25 | K=200 | K=∞ |
| --- | --- | --- | --- | --- |
| J | −5.80 | −4.11 | −4.54 | −3.79 |
| Fourier | −6.15 | −2.95 | −3.35 | −4.34 |
| h5 | −5.74 | −4.23 | −3.69 | −4.91 |
| h128 | −5.59 | 0.30 | 3.93 | 4.31 |

Solo h128 con molti layout impara una buona policy deterministica. Per J la competenza sta nella forma della distribuzione (una camminata casuale orientata), non nella sua moda. Per una policy con readout di Born il campionamento è anche l'unica valutazione fisica: ogni azione è una misura.

## 8. Equivalenza funzionale tra J e Fourier

J e Fourier si comportano tanto diversamente quanto due seed della stessa testa, e la classe di Fourier rappresenta le policy di J meglio del contrario. Le analisi sono offline, sui pesi finali (round 50) dei 10 seed a K=25 e K=200; K=200 è la cella in cui i ritorni sono più vicini.

### Distanza tra le policy

Per ogni run si raccolgono 2000 stati dai suoi episodi di test. La distanza tra due run è la distanza di variazione totale (TV, da 0 = identiche a 1 = disgiunte) tra le loro distribuzioni d'azione, mediata sugli stati di entrambi. Seed diversi significano anche pool di training diversi, da cui la tabella a due ingressi.

**Distanza di variazione totale.** Per due distribuzioni d'azione p e q nello stesso stato, **TV(p, q) = ½ · Σₐ |p(a) − q(a)|**, sommando sulle 8 azioni. Vale 0 se le due policy danno la stessa distribuzione e 1 se mettono probabilità su azioni completamente diverse; in mezzo è la quantità di probabilità da spostare per trasformare una distribuzione nell'altra. La TV si calcola stato per stato e si fa la media sui 4000 stati dei due run: un valore di 0.5–0.6 significa che in uno stato tipico le due policy distribuiscono in modo diverso circa metà della probabilità.

Tabella 9. Distanza TV media tra coppie di run, con minimo e massimo tra parentesi quadre.

| coppia di run | coppie | cosa cambia tra i due run | TV a K=25 | TV a K=200 |
| --- | --- | --- | --- | --- |
| J contro J, seed diversi | 45 | solo il seed (pool, inizializzazione, flusso casuale) | 0.54 [0.33–0.68] | 0.49 [0.29–0.63] |
| Fourier contro Fourier, seed diversi | 45 | solo il seed | 0.67 [0.55–0.84] | 0.55 [0.35–0.72] |
| J contro Fourier, stesso seed | 10 | solo la testa | 0.60 [0.51–0.73] | 0.47 [0.38–0.65] |
| J contro Fourier, seed diversi | 90 | la testa e il seed | 0.61 [0.44–0.82] | 0.53 [0.35–0.67] |
| **effetto della testa** |  |  | **+0.007** | **+0.008** |

**Come è costruita.** Per ogni K ci sono 20 run (J e Fourier, seed 0–9). Per ogni coppia di run si calcola la TV media sugli stati di entrambi; le coppie si raggruppano secondo cosa hanno in comune: 45 = coppie possibili tra 10 run della stessa testa (10·9/2), 90 = combinazioni di un J e un Fourier con seed diversi (10·9). Per ogni riga si riporta la media delle TV e, tra parentesi, la minima e la massima.

**Effetto della testa** = (J contro Fourier, seed diversi) − media di (J contro J) e (Fourier contro Fourier); a K=25: 0.61 − (0.54 + 0.67)/2 ≈ +0.007. La logica: due run della stessa testa con seed diversi differiscono già molto. Se J e Fourier calcolassero policy di tipo diverso, una coppia J–Fourier con seed diversi dovrebbe distare di più, perché la differenza di testa si aggiungerebbe a quella di seed; invece dista praticamente uguale, quindi l'identità della testa non aggiunge distanza. Le distanze sono comunque grandi in assoluto: il problema ammette molte policy diverse con lo stesso ritorno. Per l'effetto della testa non è calcolato un intervallo di confidenza: va letto rispetto alla dispersione tra coppie di run, che va da 0.29 a 0.84.

La riga "stesso seed" è un controllo a parte: le due policy condividono pool e inizializzazione dell'encoder e differiscono solo nella testa. A K=200 sono un po' più vicine (0.47 contro 0.53): pool e inizializzazione orientano entrambe le teste verso policy simili.

### Cross-distillazione

Si congela l'encoder di un run maestro e si addestra da zero una testa studente a imitarne la policy (KL), su 20 000 stati dei suoi episodi in layout nuovi. Frazione spiegata = 1 − KL(maestro‖studente) / KL(maestro‖distribuzione marginale), 1 = imitazione perfetta. Media su 10 maestri, 3 inizializzazioni ciascuno, learning rate scelto sulla validazione tra 10⁻³, 3·10⁻³ e 10⁻². MLP64 è un MLP Linear(12 → 64) – ReLU – Linear(64 → 8) da 1352 parametri, usato come controllo di capacità. Riferimento = studente della stessa testa del maestro.

Tabella 10. Cross-distillazione: frazione spiegata e ritorno di test dello studente, con quello del maestro tra parentesi (pesi finali).

| maestro | studente | frazione K=25 | ritorno studente K=25 (maestro) | frazione K=200 | ritorno studente K=200 (maestro) |
| --- | --- | --- | --- | --- | --- |
| Fourier | Fourier (riferimento) | 0.986 | 3.81 (3.83) | 0.991 | 4.47 (4.55) |
| Fourier | **J** | **0.810** | 3.87 (3.83) | **0.871** | 4.23 (4.55) |
| Fourier | MLP64 | 0.995 | 3.82 (3.83) | 0.996 | 4.52 (4.55) |
| J | J (riferimento) | 0.956 | 4.16 (4.18) | 0.958 | 4.30 (4.43) |
| J | **Fourier** | **0.881** | 4.05 (4.18) | **0.895** | 4.30 (4.43) |
| J | MLP64 | 0.967 | 4.21 (4.18) | 0.971 | 4.36 (4.43) |

Con un maestro Fourier, uno studente J resta 0.12–0.18 sotto il riferimento; con un maestro J, uno studente Fourier resta 0.06–0.08 sotto il riferimento. Il riferimento non è un tetto: MLP64 lo supera con entrambi i maestri, quindi misura quanto bene una testa imita la propria classe, non il massimo possibile. Un controllo con learning rate più alto ha cambiato le frazioni di 0.01–0.02, quindi l'asimmetria è una proprietà delle due classi di funzioni e non un limite di ottimizzazione. In ritorno la differenza conta poco: ogni studente resta entro 0.3 dal suo maestro.

**Conclusione di questa sezione.** Non emerge nulla che J calcoli e che una testa Fourier o un MLP più grande (MLP64) non riescano ad approssimare; un MLP della taglia di J (h5) non è stato provato come studente. Il VQC è un modo di ottenere un bias induttivo di tipo Fourier con pochi parametri.

## 9. Robustezza al rumore su hardware IBM simulato

Con il rumore dei processori IBM attuali la policy di J perde 0.10 di ritorno su Torino (nessuna differenza rilevata) e 0.38 su Brisbane (9%), restando sopra la testa h5 senza rumore e al livello di Fourier senza rumore.

Si valutano le 10 policy di J addestrate senza rumore a K=25 (pesi finali: senza rumore J vale 4.18 invece del 4.05 della Tabella 4), eseguite con rumore sulle porte del circuito, con la stessa valutazione di test (100 layout nuovi, azioni campionate). Il training sotto rumore non è incluso.

### Tre livelli di realismo

1. **Rumore sintetico:** canale depolarizzante con probabilità p2 dopo ogni CNOT e p1 = p2/10 dopo ogni rotazione, p2 ∈ {0, 0.1, 0.3, 1, 3}%. Sono valori scelti da noi, non legati a nessuna macchina: servono a vedere come cala il ritorno al crescere del rumore. A parte si è provato il solo errore di lettura, sempre scelto da noi: ogni bit misurato si inverte con probabilità 2%, le porte sono perfette. Il ritorno è 4.27 contro 4.18 senza rumore (differenza appaiata +0.09, CI [−0.09, 0.27]): nessuna differenza rilevata.
2. **Modello calibrato effettivo:** il circuito è transpilato sui backend simulati di IBM `fake_torino` (processore Heron) e `fake_brisbane` (Eagle), che contengono topologia, porte native e una fotografia delle calibrazioni reali. Dalle calibrazioni dei 3 qubit fisici scelti e dal numero di porte transpilate (42 porte a 2 qubit) si ricavano p1, p2 e l'errore di lettura effettivi: p2 = 0.9% su Torino e 1.8% su Brisbane (derivazione nell'Appendice A).
3. **Simulazione Aer completa:** ogni azione di ogni episodio viene da **un solo shot** del circuito transpilato, simulato con il modello di rumore completo del backend (errori per singolo qubit e coppia, rilassamento T1/T2 con la durata delle porte, errore di lettura). È ciò che farebbe l'hardware.

[embedded content: Figura 4 · ritorno di J sotto rumore · scripts/eval_noise.py e scripts/eval_aer.py · J a K=25, pesi finali, 10 seed, 100 episodi di test per punto]

In Figura 4 i due backend reali cadono sulla curva del rumore sintetico, vicino al tratto in cui il ritorno comincia a scendere; il degrado diventa netto solo a p2 = 3%.

### Backend calibrati

Tabella 11. **Ritorno di test di J** (media ± SE su 10 seed, stessi pesi e stessi 100 layout in ogni riga):

| simulazione | IBM Torino | IBM Brisbane |
| --- | --- | --- |
| NR: senza rumore | 4.18 ± 0.44 | 4.18 ± 0.44 |
| MDE: modello depolarizzante effettivo | 4.11 ± 0.42 | 3.89 ± 0.41 |
| AER: simulazione Aer completa | 4.09 ± 0.38 | 3.80 ± 0.41 |

Tabella 12. **Differenze appaiate per seed** (media e CI al 95%):

| confronto | cosa misura | IBM Torino | IBM Brisbane |
| --- | --- | --- | --- |
| AER − NR | quanto ritorno costa il rumore della macchina | −0.10 [−0.28, 0.09] | **−0.38 [−0.62, −0.14]** |
| AER − MDE | quanto sbaglia il modello semplificato | −0.03 [−0.22, 0.17] | −0.09 [−0.26, 0.07] |

In grassetto l'unico CI che esclude 0.

### Lettura

- **L'hardware attuale non rompe la policy.** Su Torino la perdita non si rileva; su Brisbane è del 9%. In entrambi i casi J resta molto sopra h5 a pari entropia senza rumore (2.06, pesi finali) e al livello di Fourier senza rumore (3.83, pesi finali).
- **Il degrado è graduale.** Fino a p2 = 1% il ritorno non cambia oltre il rumore della valutazione; diventa netto solo a p2 = 3% (−0.79, −19%), circa il doppio dell'errore di Brisbane.
- **Il modello effettivo è adeguato.** Non si rileva differenza tra il modello semplificato e la simulazione Aer completa (CI stretti attorno a 0).
- **Costo dell'esecuzione su hardware:** circa 7000 circuiti per valutazione (100 episodi), 1 shot ciascuno; i passi dipendono dall'azione precedente, quindi si raggruppano gli episodi (un job per passo) ma non i passi.

## 10. Conclusioni, limiti e passi successivi

Esiste un regime in cui la testa VQC supera l'MLP della stessa taglia, pochi layout di training, e la ragione è un bias induttivo di tipo Fourier che il circuito ottiene con 117 parametri; non è emerso un vantaggio specifico del circuito.

### Cosa si può affermare

1. A K=25 J generalizza meglio dell'MLP della stessa taglia anche a pari entropia (+2.13, CI [0.82, 3.44], 10 seed) e quanto un MLP 23 volte più grande regolarizzato alla stessa entropia.
2. Il vantaggio dipende da K: c'è con 5 e 25 layout, non si rileva da 200 in su, dove l'MLP grande è migliore.
3. A K=5 nessuna testa generalizza e il vantaggio di J deriva in gran parte dalla sua stocasticità in valutazione.
4. Una testa classica con feature di Fourier eguaglia J in ritorno e comportamento; la classe di Fourier contiene quella di J meglio del contrario.
5. Con il rumore calibrato di processori IBM attuali la policy perde tra 0 e 9% del ritorno.

### Limiti

- **Potenza statistica:** con 10 seed J contro Fourier a K=25 ha un CI largo ([−0.82, 1.56]); una piccola differenza in un verso o nell'altro non è esclusa. A K=5 resta un residuo di circa +1 a favore di J non risolto.
- **Learning rate:** J usa un lr scelto con una scansione, le teste MLP il default di A2C senza taratura; per Fourier il lr di J peggiora il risultato.
- **Un solo ambiente e un solo circuito:** i risultati valgono per questo POMDP e per il circuito J; un altro ansatz provato (K, 131 parametri) non ha dato guadagni (3.45 contro 4.05 a K=25).
- **Rumore:** applicato in esecuzione a policy addestrate senza rumore, su backend simulati con calibrazioni fotografate; niente derive nel tempo né crosstalk.
- **Encoder classico:** le feature sono prodotte da un encoder classico addestrato insieme alla testa; il confronto riguarda solo la testa.

### Passi successivi

- **Hardware reale IBM:** eseguire la policy di J su un processore reale, con lo stesso codice della simulazione Aer completa, e confrontare il ritorno con la simulazione calibrata dello stesso giorno. Si può partire da un seed e pochi episodi, dato il costo in code (un job per passo).
- **Più seed per J contro Fourier a K=25**, se serve affermare una differenza di ritorno tra le due teste.
- **Training sotto rumore**, se interessa la robustezza dell'apprendimento oltre a quella dell'esecuzione.
