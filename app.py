import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Configurazione pagina
st.set_page_config(
    page_title="Meccanica del Gradient Boosting Regressor",
    page_icon="📊",
    layout="wide"
)

# Titolo e introduzione
st.title("Comprensione della Meccanica del Gradient Boosting Regressor")
st.markdown("""
Questa app spiega passo per passo il funzionamento dell'algoritmo Gradient Boosting Regressor (GBR), 
utilizzando un esempio semplificato con pochi dati e solo due variabili.
""")

# Creiamo un dataset semplice direttamente nell'app
st.header("1. Creazione del Dataset di Esempio")

st.markdown("""
Per comprendere meglio l'algoritmo, utilizzeremo un dataset molto semplice con solo 10 esempi e 2 variabili:
- **Età**: età della persona (da 20 a 65 anni)
- **Fumatore**: se la persona è fumatore (1) o non fumatore (0)

L'obiettivo è predire il **costo assicurativo** in base a queste due variabili.
""")

# Permetti all'utente di vedere il codice che genera i dati
with st.expander("Mostra il codice per generare i dati"):
    st.code("""
# Generiamo un dataset semplice con 10 esempi
np.random.seed(42)  # Per riproducibilità
n_samples = 10
ages = np.random.randint(20, 65, n_samples)
smoker = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# Formula per il costo assicurativo:
# base_cost + age_factor*age + smoker_factor*smoker + rumore_casuale
base_cost = 5000
age_factor = 50
smoker_factor = 8000
noise = np.random.normal(0, 1000, n_samples)

insurance_cost = base_cost + age_factor * ages + smoker_factor * smoker + noise
""", language="python")

# Generazione dati
np.random.seed(42)
n_samples = 10
ages = np.random.randint(20, 65, n_samples)
smoker = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# Formula per generare il target
base_cost = 5000
age_factor = 50
smoker_factor = 8000
noise = np.random.normal(0, 1000, n_samples)
insurance_cost = base_cost + age_factor * ages + smoker_factor * smoker + noise

# Dataframe
data = pd.DataFrame({
    'età': ages,
    'fumatore': smoker,
    'costo_assicurativo': insurance_cost
})

# Mostra dataset
st.subheader("Il nostro dataset:")
st.dataframe(data, use_container_width=True)

# Visualizzazione dei dati
st.subheader("Visualizzazione dei dati")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(data['età'], data['costo_assicurativo'], c=['red' if x == 1 else 'blue' for x in data['fumatore']], 
               s=100, alpha=0.7)
    ax1.set_xlabel('Età')
    ax1.set_ylabel('Costo Assicurativo')
    ax1.set_title('Relazione tra Età e Costo Assicurativo')
    ax1.grid(True)
    # Aggiungi legenda
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Fumatore')
    blue_patch = mpatches.Patch(color='blue', label='Non Fumatore')
    ax1.legend(handles=[red_patch, blue_patch])
    st.pyplot(fig1)

with col2:
    st.markdown("""
    **Osservazioni sul dataset:**
    
    1. I fumatori (punti rossi) tendono ad avere costi assicurativi molto più alti
    2. C'è una relazione lineare tra età e costo (aumenta di circa 50 per anno di età)
    3. Il costo base è circa 5000
    4. C'è del rumore casuale nei dati
    
    Questo dataset è stato generato con una formula nota:
    ```
    costo = 5000 + 50*età + 8000*fumatore + rumore_casuale
    ```
    
    Ma in un caso reale non conosceremmo questa formula e dovremmo scoprirla dai dati.
    """)

# Prepara i dati per il modello
X = data[['età', 'fumatore']]
y = data['costo_assicurativo']

# Spiegazione del GBR
st.header("2. Comprensione del Gradient Boosting Regressor")

st.markdown("""
Il Gradient Boosting è una tecnica di ensemble learning che costruisce modelli in modo sequenziale, 
dove ogni nuovo modello cerca di correggere gli errori dei modelli precedenti.

Vediamo passo per passo le prime 3 iterazioni del GBR:
""")

# Passo 1: Inizializzazione
st.subheader("Passo 1: Inizializzazione con la media")

initial_prediction = np.mean(y)
residuals = y - initial_prediction

step1_df = pd.DataFrame({
    'età': data['età'],
    'fumatore': data['fumatore'],
    'valore_reale': y,
    'predizione_iniziale': np.ones(n_samples) * initial_prediction,
    'residuo': residuals
})

st.markdown(f"""
Il GBR inizia con un modello molto semplice: predice lo stesso valore per tutti i punti.
Questo valore iniziale è tipicamente la **media** del target: **{initial_prediction:.2f}**
""")

st.dataframe(step1_df, use_container_width=True)

# Visualizza residui
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(range(n_samples), residuals, c='green', s=100, alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='-')
ax2.set_xlabel('Indice del campione')
ax2.set_ylabel('Residuo')
ax2.set_title('Residui dopo l\'inizializzazione')
ax2.grid(True)
st.pyplot(fig2)

st.markdown("""
I **residui** sono la differenza tra i valori reali e le predizioni. 
Valori positivi significano che stiamo sottostimando il costo, valori negativi che lo stiamo sovrastimando.

Osserviamo che:
- I residui per i fumatori tendono ad essere molto positivi (stiamo sottostimando il loro costo)
- I residui variano anche in base all'età
""")

# Passo 2: Primo albero
st.subheader("Passo 2: Addestramento del primo albero decisionale")

# Imposta learning rate
learning_rate = 0.1
tree_depth = 1  # Albero molto semplice per comprensibilità

# Addestra il primo albero sui residui
tree1 = DecisionTreeRegressor(max_depth=tree_depth)
tree1.fit(X, residuals)

# Predizioni del primo albero
tree1_predictions = tree1.predict(X)

# Aggiorna le predizioni
updated_predictions1 = initial_prediction + learning_rate * tree1_predictions
new_residuals1 = y - updated_predictions1

# Mostra risultati
step2_df = pd.DataFrame({
    'età': data['età'],
    'fumatore': data['fumatore'],
    'valore_reale': y,
    'residuo_precedente': residuals,
    'predizione_albero1': tree1_predictions,
    'contributo_albero1': learning_rate * tree1_predictions,
    'predizione_aggiornata': updated_predictions1,
    'nuovo_residuo': new_residuals1
})

st.markdown(f"""
Addestriamo un **albero decisionale** per prevedere i residui (non i valori originali!).
Usiamo un albero molto semplice (profondità=1) e un learning rate basso ({learning_rate}) per
limitare il contributo di ogni singolo albero.
""")

st.dataframe(step2_df, use_container_width=True)

# Mostra cosa ha imparato l'albero
st.markdown("### Cosa ha imparato il primo albero?")

# Estraiamo la struttura dell'albero
tree_structure = []
if tree1.tree_.feature[0] == 0:  # Età
    feature_name = "età"
    threshold = tree1.tree_.threshold[0]
else:  # Fumatore
    feature_name = "fumatore"
    threshold = tree1.tree_.threshold[0]

left_value = tree1.tree_.value[1][0][0]
right_value = tree1.tree_.value[2][0][0]

st.markdown(f"""
L'albero decisionale ha imparato a dividere i dati in base a: **{feature_name}**

La regola è:
- Se {feature_name} <= {threshold:.2f}: predici {left_value:.2f}
- Se {feature_name} > {threshold:.2f}: predici {right_value:.2f}

Questo albero ci sta dicendo che la variabile più importante per prevedere i residui è **{feature_name}**.
""")

# Visualizza l'effetto del primo albero
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(range(n_samples), residuals, label='Residui originali', c='green', s=80, alpha=0.7)
ax3.scatter(range(n_samples), new_residuals1, label='Residui dopo albero 1', c='orange', s=80, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='-')
ax3.set_xlabel('Indice del campione')
ax3.set_ylabel('Residuo')
ax3.set_title('Residui prima e dopo il primo albero')
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

st.markdown("""
Osserviamo che i residui si sono ridotti, ma sono ancora lontani da zero.
Il primo albero ha catturato parte del pattern, ma non tutto.
""")

# Passo 3: Secondo albero
st.subheader("Passo 3: Addestramento del secondo albero decisionale")

# Addestra il secondo albero sui nuovi residui
tree2 = DecisionTreeRegressor(max_depth=tree_depth)
tree2.fit(X, new_residuals1)

# Predizioni del secondo albero
tree2_predictions = tree2.predict(X)

# Aggiorna le predizioni
updated_predictions2 = updated_predictions1 + learning_rate * tree2_predictions
new_residuals2 = y - updated_predictions2

# Mostra risultati
step3_df = pd.DataFrame({
    'età': data['età'],
    'fumatore': data['fumatore'],
    'valore_reale': y,
    'residuo_precedente': new_residuals1,
    'predizione_albero2': tree2_predictions,
    'contributo_albero2': learning_rate * tree2_predictions,
    'predizione_aggiornata': updated_predictions2,
    'nuovo_residuo': new_residuals2
})

st.markdown("""
Addestriamo un **secondo albero decisionale** per prevedere i nuovi residui.
Questo albero cercherà di catturare i pattern che il primo albero ha mancato.
""")

st.dataframe(step3_df, use_container_width=True)

# Mostra cosa ha imparato il secondo albero
st.markdown("### Cosa ha imparato il secondo albero?")

# Estraiamo la struttura dell'albero
if tree2.tree_.feature[0] == 0:  # Età
    feature_name2 = "età"
    threshold2 = tree2.tree_.threshold[0]
else:  # Fumatore
    feature_name2 = "fumatore"
    threshold2 = tree2.tree_.threshold[0]

left_value2 = tree2.tree_.value[1][0][0]
right_value2 = tree2.tree_.value[2][0][0]

st.markdown(f"""
L'albero decisionale ha imparato a dividere i dati in base a: **{feature_name2}**

La regola è:
- Se {feature_name2} <= {threshold2:.2f}: predici {left_value2:.2f}
- Se {feature_name2} > {threshold2:.2f}: predici {right_value2:.2f}

Questo secondo albero sta cercando di catturare un pattern diverso rispetto al primo.
""")

# Visualizza l'effetto del secondo albero
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.scatter(range(n_samples), residuals, label='Residui originali', c='green', s=80, alpha=0.5)
ax4.scatter(range(n_samples), new_residuals1, label='Residui dopo albero 1', c='orange', s=80, alpha=0.5)
ax4.scatter(range(n_samples), new_residuals2, label='Residui dopo albero 2', c='blue', s=80, alpha=0.7)
ax4.axhline(y=0, color='r', linestyle='-')
ax4.set_xlabel('Indice del campione')
ax4.set_ylabel('Residuo')
ax4.set_title('Evoluzione dei residui')
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

st.markdown("""
Notiamo che i residui continuano a ridursi, avvicinandosi sempre più a zero.
""")

# Passo 4: Terzo albero
st.subheader("Passo 4: Addestramento del terzo albero decisionale")

# Addestra il terzo albero sui nuovi residui
tree3 = DecisionTreeRegressor(max_depth=tree_depth)
tree3.fit(X, new_residuals2)

# Predizioni del terzo albero
tree3_predictions = tree3.predict(X)

# Aggiorna le predizioni
updated_predictions3 = updated_predictions2 + learning_rate * tree3_predictions
new_residuals3 = y - updated_predictions3

# Mostra risultati
step4_df = pd.DataFrame({
    'età': data['età'],
    'fumatore': data['fumatore'],
    'valore_reale': y,
    'residuo_precedente': new_residuals2,
    'predizione_albero3': tree3_predictions,
    'contributo_albero3': learning_rate * tree3_predictions,
    'predizione_finale': updated_predictions3,
    'residuo_finale': new_residuals3
})

st.markdown("""
Addestriamo un **terzo albero decisionale** per prevedere i residui rimasti dopo i primi due alberi.
""")

st.dataframe(step4_df, use_container_width=True)

# Visualizza l'evoluzione dei residui
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.scatter(range(n_samples), residuals, label='Residui originali', c='green', s=80, alpha=0.4)
ax5.scatter(range(n_samples), new_residuals1, label='Dopo albero 1', c='orange', s=80, alpha=0.4)
ax5.scatter(range(n_samples), new_residuals2, label='Dopo albero 2', c='blue', s=80, alpha=0.4)
ax5.scatter(range(n_samples), new_residuals3, label='Dopo albero 3', c='red', s=80, alpha=0.7)
ax5.axhline(y=0, color='r', linestyle='-')
ax5.set_xlabel('Indice del campione')
ax5.set_ylabel('Residuo')
ax5.set_title('Evoluzione dei residui nei tre passaggi')
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# Calcolo dell'errore ad ogni iterazione
mse_initial = mean_squared_error(y, np.ones(n_samples) * initial_prediction)
mse_tree1 = mean_squared_error(y, updated_predictions1)
mse_tree2 = mean_squared_error(y, updated_predictions2)
mse_tree3 = mean_squared_error(y, updated_predictions3)

# Visualizza diminuzione dell'errore
st.subheader("Riduzione dell'errore nelle iterazioni")

error_df = pd.DataFrame({
    'Iterazione': ['Iniziale', 'Dopo albero 1', 'Dopo albero 2', 'Dopo albero 3'],
    'MSE': [mse_initial, mse_tree1, mse_tree2, mse_tree3],
    'Riduzione %': [0, (1-mse_tree1/mse_initial)*100, (1-mse_tree2/mse_initial)*100, (1-mse_tree3/mse_initial)*100]
})

st.dataframe(error_df, use_container_width=True)

# Grafico della riduzione dell'errore
fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.plot(['Iniziale', 'Albero 1', 'Albero 2', 'Albero 3'], 
        [mse_initial, mse_tree1, mse_tree2, mse_tree3], 
        'bo-', linewidth=2, markersize=10)
ax6.set_xlabel('Iterazione')
ax6.set_ylabel('Errore quadratico medio (MSE)')
ax6.set_title('Diminuzione dell\'errore nelle iterazioni')
ax6.grid(True)
st.pyplot(fig6)

# Confronto predizioni vs valori reali
st.subheader("Confronto tra valori reali e predizioni finali")

fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.scatter(range(n_samples), y, label='Valori reali', c='blue', s=100, alpha=0.7)
ax7.scatter(range(n_samples), updated_predictions3, label='Predizioni finali', c='red', s=100, alpha=0.7)
ax7.set_xlabel('Indice del campione')
ax7.set_ylabel('Costo assicurativo')
ax7.set_title('Confronto tra valori reali e predizioni finali')
ax7.legend()
ax7.grid(True)
st.pyplot(fig7)

# Formula generale del GBR
st.header("3. Formula matematica del Gradient Boosting")

st.markdown("""
La formula generale del Gradient Boosting Regressor è:

1. **Inizializzazione**: $F_0(x) = \mathrm{media}(y)$

2. **Per ogni iterazione** $m = 1, 2, ..., M$:
   - Calcolare i residui: $r_i = y_i - F_{m-1}(x_i)$
   - Addestrare un albero $h_m(x)$ sui residui
   - Aggiornare il modello: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$

dove $\eta$ è il learning rate (nel nostro esempio: 0.1).

Nel nostro caso, dopo 3 iterazioni:

$F_3(x) = \mathrm{media}(y) + \eta \cdot [h_1(x) + h_2(x) + h_3(x)]$

$F_3(x) = {initial_prediction:.2f} + {learning_rate} \cdot [h_1(x) + h_2(x) + h_3(x)]$
""")

# Sezione conclusioni
st.header("4. Conclusioni e considerazioni finali")

st.markdown("""
### Cosa abbiamo imparato sul Gradient Boosting

1. **Apprendimento sequenziale**: Il GBR costruisce alberi in sequenza, dove ogni albero impara dai "errori" dei precedenti.

2. **Residui chiave**: Invece di prevedere direttamente i valori target, ogni nuovo albero prevede i residui.

3. **Learning rate**: Il parametro $\eta$ controlla quanto "pesa" il contributo di ogni nuovo albero.

4. **Miglioramento graduale**: Ad ogni iterazione, l'errore diminuisce e le previsioni si avvicinano ai valori reali.

5. **Capacità adattiva**: Il modello impara automaticamente quali feature sono importanti e come utilizzarle.

### Vantaggi del GBR:

- **Alta accuratezza**: Spesso tra i migliori algoritmi per problemi di regressione
- **Flessibilità**: Cattura relazioni non lineari e interazioni tra variabili
- **Robustezza**: Resistente agli outlier e ai valori mancanti
- **Interpretabilità**: Fornisce l'importanza delle feature

Nei modelli reali, si utilizzano molti più alberi (tipicamente 100-1000) e dati per ottenere previsioni accurate.
""")

# Riferimenti
st.markdown("""
---
### Riferimenti
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
- Scikit-learn documentation: [Gradient Boosting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
""")
