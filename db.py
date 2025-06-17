import streamlit as st
import random
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import streamviz

df = pd.read_csv("df_sample_full.csv")
all_id = sorted(df['SK_ID_CURR'].unique())


cols_exclues = ['TARGET','Unnamed: 0','SK_ID_CURR']
df_w_target = df[df['TARGET'].notnull()]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(method="ffill")
df_0 = df[df["TARGET"] == 0]
df_1 = df[df["TARGET"] == 1]
features = [col for col in df_w_target if col not in cols_exclues] #Pour éviter d'afficher la distribution de la colonne TARGET

df_shap_all = pd.read_csv('shap_summary_all.csv')
df_shap_obs = pd.read_csv('shap_values_all_data.csv')

seuil = 0.5





# URL de l'API
url = "https://api7oc-g3a0e5b8fjfbcwcy.canadacentral-01.azurewebsites.net/predict"

# Nouvelle page pour afficher la Comparaison de clients et distribution des variables
st.sidebar.title("Analyse des Variables")
page = st.sidebar.radio("Sélectionnez une page", ["Prédiction", "Distribution comparée des clients fiables et défaillants"])


# Fonction de prédiction
def get_prediction(test_sample_dict):
    # Envoyer la requête à l'API
    response = requests.post(url, json=test_sample_dict)
    
    if response.status_code == 200:
        # Si la réponse est correcte, on renvoie le score
        probabilité = round(response.json().get('prediction', [None])[0], 3)
        
        
        return probabilité
    else:
        return None


# Menu déroulant pour sélectionner le SK_ID_CURR
sk_id_curr = st.selectbox("Sélectionnez le SK_ID_CURR du client", all_id )

# Rechercher la ligne correspondant à SK_ID_CURR
test_sample = df[df['SK_ID_CURR'] == sk_id_curr].copy()

# Enlever la colonne 'TARGET' et nettoyer la ligne
test_sample = test_sample.drop('TARGET', axis=1)
test_sample.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

if page == "Prédiction" :

    

    # Récupérer la ligne du client (par exemple client 1)
    selected_row = df[df["SK_ID_CURR"] == sk_id_curr].copy().drop("Unnamed: 0",axis=1)
    test_sample_dict = test_sample.to_dict(orient='records')[0]
    score_non_changed = get_prediction(test_sample_dict)
    st.write(f"Probabilité de défaillance du client {sk_id_curr} : **{score_non_changed}**")
    if score_non_changed >= seuil:
        st.write("**Ce client est défaillant.**")
    else:
        st.write("**Ce client est fiable.**")  

    #st.dataframe(selected_row)
    st.subheader("Modifiez les valeurs pour afficher une nouvelle probabilité...")
    
    # Afficher les valeurs actuelles des variables 
    current_gender = test_sample['CODE_GENDER'].values[0]
    current_ext_source_1 = test_sample['EXT_SOURCE_1'].values[0]
    current_ext_source_2 = test_sample['EXT_SOURCE_2'].values[0]
    current_ext_source_3 = test_sample['EXT_SOURCE_3'].values[0]

    col1, col2, col3, col4 = st.columns(4)

    # Ajouter des curseurs pour modifier les valeurs des variables
    
    with col1 :
        new_gender = st.slider("CODE_GENDER", min_value=0, max_value=1, value=current_gender)
    with col2 :
        new_ext_source_1 = st.slider(f"EXT_SOURCE_1 ({round(current_ext_source_1,3)})", 
                                    min_value=0.0, max_value=1.0, step=0.1, 
                                    value=current_ext_source_1)
    with col3 :
        new_ext_source_2 = st.slider(f"EXT_SOURCE_2 ({round(current_ext_source_2,3)})", 
                                    min_value=0.0, max_value=1.0, step=0.1, 
                                    value=current_ext_source_2)
    with col4 :
        new_ext_source_3 = st.slider(f"EXT_SOURCE_3 ({round(current_ext_source_3,3)})", 
                                    min_value=0.0, max_value=1.0, step=0.1, 
                                    value=current_ext_source_3)
                                    


    # Modifier les valeurs dans la ligne du DataFrame
    test_sample['CODE_GENDER'] = new_gender
    test_sample['EXT_SOURCE_1'] = new_ext_source_1
    test_sample['EXT_SOURCE_2'] = new_ext_source_2
    test_sample['EXT_SOURCE_3'] = new_ext_source_3

    # Convertir la ligne en dictionnaire
    test_sample_dict = test_sample.to_dict(orient='records')[0]

    # Nouvelle prédiction à chaque changement
    new_probabilité = get_prediction(test_sample_dict)

    if new_probabilité is not None:
        st.subheader(f"Nouvelle probabilité de défaillance (après modification) : **{round(new_probabilité,2)}**")
    else:
        st.error("Erreur lors de la nouvelle prédiction.")

    # Afficher la jauge mise à jour avec la nouvelle valeur
    streamviz.gauge(new_probabilité, gMode = "gauge", sFix="%", gcHigh="#E69F00", gcMid="#56B4E9", gcLow="#009E73", gSize="MED")
    if new_probabilité >= 0.7:
        statut = "🔴 Risque élevé"
    elif new_probabilité >= 0.4:
        statut = "🟡 Risque modéré"
    else:
        statut = "🟢 Risque faible"

    st.markdown(f"**Statut du client : {statut}**")


    st.subheader("Importance des variables dans la prédiction :")

    shap_obs = df_shap_obs[df_shap_obs["SK_ID_CURR"]==sk_id_curr].transpose()
    shap_obs = shap_obs.reset_index()
    shap_obs.columns = ['feature', 'shap_importance_obs']
    full_shap = shap_obs.merge(df_shap_all, on="feature")
    

    # Melt du DataFrame pour avoir une colonne 'Value_Type' et une colonne 'Value'
    df_melted = full_shap.melt(id_vars='feature', value_vars=['shap_importance_obs', 'shap_importance'], 
                        var_name='Value_Type', value_name='Value')

    # Création du graphique
    fig1 = plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='feature', hue='Value_Type', data=df_melted, palette='viridis')

    
    plt.title('Comparaison des valeurs SHAP pour chaque feature')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.legend(title='Type de Valeur', labels=['Importance SHAP - Observée', 'Importance SHAP - Modèle'], fontsize=12, title_fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()



    st.pyplot(fig1)


    

    

    

    
 
    


if page == "Distribution comparée des clients fiables et défaillants":
    

    selected_row = df[df["SK_ID_CURR"] == sk_id_curr].copy().drop("Unnamed: 0",axis=1)
    test_sample_dict = test_sample.to_dict(orient='records')[0]
    score_non_changed = get_prediction(test_sample_dict)
    st.write(f"Probabilité de défaillance du client {sk_id_curr} : **{score_non_changed}**")
        

    # Choix d’un second client à comparer (facultatif)
    sk_id_compare = st.sidebar.selectbox("Comparer avec un autre client ?", df["SK_ID_CURR"].unique(), index=1)
    compare_row = df[df["SK_ID_CURR"] == sk_id_compare].copy().drop(["TARGET"], axis=1)
    compare_row.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
    compare_dict = compare_row.to_dict(orient='records')[0]
    

    compare_score = get_prediction(compare_dict)
    if compare_score is not None:
        st.write(f"Probabilité de défaillance du client {sk_id_compare} : **{compare_score:.2f}**")
        
    else:
        st.warning(f"Impossible de calculer la probabilité pour le client {sk_id_compare}")   
    
        
    

    # Sélection de la variable à afficher
    selected_col = st.sidebar.selectbox("Choisissez une variable :", features)

    # Vérifie et convertit les booléens en entiers si besoin
    if df_0[selected_col].dtype == 'bool':
        df_0[selected_col] = df_0[selected_col].astype(int)
        df_1[selected_col] = df_1[selected_col].astype(int)

    # Nettoyage des données
    data_0 = df_0[selected_col].replace([np.inf, -np.inf], np.nan).dropna()
    data_1 = df_1[selected_col].replace([np.inf, -np.inf], np.nan).dropna()

    selected_value = test_sample_dict.get(selected_col, None)
    selected_value_2 = compare_dict.get(selected_col,None)

    # Couleurs adaptées au daltonisme
    couleur_fiable = '#0072B2'     # bleu-violet
    couleur_defaillant = '#E69F00' # orange clair
    couleur_client = '#009E73'     # vert
    couleur_client_2 = '#999999'   # gris foncé

    # Création du graphique
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=data_0, label='Distribution des clients fiables', color=couleur_fiable, ax=ax)
    sns.kdeplot(data=data_1, label='Distribution des clients défaillants', color=couleur_defaillant, ax=ax)

    if pd.notna(selected_value) and np.isfinite(selected_value):
        ax.axvline(x=selected_value, color='green', linestyle='--', linewidth=2, label=f'Client {sk_id_curr}')
        ax.text(selected_value, ax.get_ylim()[1] * 0.9, f"{selected_value:.2f}", color=couleur_client, rotation=90)

    if pd.notna(selected_value_2) and np.isfinite(selected_value_2):
        ax.axvline(x=selected_value_2, color='orange', linestyle='--', linewidth=2, label=f'Client {sk_id_curr}')
        ax.text(selected_value_2, ax.get_ylim()[1] * 0.9, f"{selected_value_2:.2f}", color=couleur_client_2, rotation=90)

    ax.set_title(f'Distribution de "{selected_col}"')

    ax.set_title(f'Distribution de "{selected_col}"')
    ax.legend()

    st.pyplot(fig)

    # Calcul des statistiques
    def describe_stats(data):
        return {
            "Moyenne": np.mean(data),
            "Médiane": np.median(data),
            "Min": np.min(data),
            "Max": np.max(data)
        }

    stats_0 = describe_stats(data_0)
    stats_1 = describe_stats(data_1)

    # Affichage des stats dans un tableau
    st.subheader("Statistiques descriptives")
    stats_df = pd.DataFrame([stats_0, stats_1], index=["Clients fiables", "Clients défaillants"])
    st.table(stats_df)