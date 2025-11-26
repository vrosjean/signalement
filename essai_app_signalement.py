import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import StringIO
import traceback # <-- Importer pour un meilleur traceback

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Analyse Signalements RATP",
    page_icon="🚇",
    layout="wide"
)

# --- Couleurs (inspirées RATP) ---
RATP_GREEN = "#00a281"
RATP_BLUE = "#0064a0"

# --- MODIFICATION : Mots-clés pour la classification de sécurité ---
# Dictionnaire enrichi avec plus de synonymes et de termes.
KEYWORDS_SECURITE = {
    "Agression / Violence": [
        'agression', 'violent', 'frappé', 'battu', 'violence', 'coup', 'menace', 
        'bagarre', 'rixe', 'insulté', 'poussé', 'bousculé', 'menacé', 'agressif',
        'altercation', 'gifle', 'crachat' # Crachat est aussi une agression
    ],
    "Harcèlement / Sexisme": [
        'harcèlement', 'harcelé', 'frottement', 'exhibition', 'sexiste', 'insultes', 
        'outrage', 'mains aux fesses', 'comportement inapproprié', 'frotteur', 
        'exhibitionniste', 'remarques', 'sexuel', 'attouchements', 'obscène',
        'propos sexistes', 'gestes déplacés'
    ],
    "Malaise / Assistance": [
        'malaise', 'tombé', 'chute', 'blessé', 'urgence', 'assistance', 'personne au sol', 
        'sdf', 'évanoui', 'secours', 'aide', 'blessure', 'urgence médicale', 
        'sans abri', 'difficulté respiratoire', 'inconscient'
    ],
    "Dégradation": [
        'dégradation', 'cassé', 'fracassé', 'vandalisme', 'tag', 'graffiti', 
        'abîmé', 'détruit', 'vitre cassée', 'siège arraché', 'détérioration',
        'brisé'
    ],
    # Cette catégorie sert de "fourre-tout" par défaut
    "Incivilité / Conflit / Autre": [
        'incivilité', 'crie', 'dispute', 'impoli', 'conflit', 'fume', 
        'cigarette', 'musique forte', 'vol', 'volé', 'pickpocket', 'détroussé', 
        'voleur', 'arraché', 'urine', 'alcoolisé', 'ivre', 'criant', 'tapage', 
        'pieds sur le siège', 'mange', 'mendicité', 'sans titre', 'fraude'
    ]
}
# -----------------------------------------------------------------

# --- Chargement et Préparation des données ---
@st.cache_data
def load_data(uploaded_file, rows_to_skip):
    """
    Charge les données depuis un fichier téléversé et effectue un nettoyage et 
    une ingénierie des caractéristiques (feature engineering) temporelles.
    """
    try:
        # MODIFICATION: Lit depuis l'objet fichier téléversé et utilise skiprows
        # --- AJOUT: Vérification du type de fichier ---
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, delimiter=';', encoding='utf-8-sig', skiprows=rows_to_skip)
        elif uploaded_file.name.endswith('.xlsx'):
            # Note: L'environnement doit avoir 'openpyxl' d'installé (ex: pip install openpyxl)
            df = pd.read_excel(uploaded_file, skiprows=rows_to_skip)
        else:
            st.error(f"Type de fichier non supporté : {uploaded_file.name}. Veuillez utiliser .csv ou .xlsx.")
            return None
        # --- FIN AJOUT ---
        
        df.columns = df.columns.str.strip() 

        date_col = None
        heure_col = None
        nature_col = None # <-- Colonne "Nature" ou "Catégorie"
        perimetre_col = None
        message_col = None 

        # Recherche insensible à la casse
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
            if col.lower() == 'heure':
                heure_col = col
            
            # --- MODIFICATION CLÉ ---
            # Recherche 'nature' OU 'catégorie' (basé sur vos fichiers)
            if col.lower() == 'nature' or col.lower() == 'catégorie':
                nature_col = col
            # --- FIN MODIFICATION CLÉ ---
                
            if col.lower() == 'périmètre': 
                perimetre_col = col
            if col.lower() == 'perimetre':
                perimetre_col = col
            if col.lower() == 'message':
                message_col = col

        if not date_col:
            with open("erreur_log.txt", "w", encoding="utf-8") as f:
                f.write("ERREUR FATALE (Code Version Corrigée) : Colonne 'Date' introuvable (même insensible à la casse).\n")
                f.write(f"Colonnes disponibles dans le CSV : {list(df.columns)}\n")
            st.error(f"Erreur: Colonne 'Date' introuvable. Vérifiez 'erreur_log.txt'.")
            return None
        
        # Colonnes 'Nature'/'Catégorie' et 'Périmètre'
        if not nature_col:
            st.warning("Colonne 'Nature' ou 'Catégorie' introuvable. Remplissage par 'Non défini'.")
            df['Nature_Clean'] = 'Non défini'
        else:
            df['Nature_Clean'] = df[nature_col].str.strip().fillna('Non défini')
            # Ne supprime la colonne que si le nom est différent (ex: 'Catégorie')
            if 'Nature_Clean' != nature_col:
                 df = df.drop(columns=[nature_col])

        if not perimetre_col:
            st.warning("Colonne 'Périmètre' introuvable. Remplissage par 'Non défini'.")
            df['Perimetre_Clean'] = 'Non défini'
        else:
            df['Perimetre_Clean'] = df[perimetre_col].str.strip().fillna('Non défini')
            if 'Perimetre_Clean' != perimetre_col:
                df = df.drop(columns=[perimetre_col])
        
        # --- MODIFICATION : Classification automatique (IA par mots-clés) ---
        if not message_col:
            st.warning("Colonne 'Message' introuvable. L'analyse de sous-catégorie ne peut pas être effectuée.")
            df['Sous_Categorie'] = 'N/A'
        else:
            # S'assurer que la colonne message est de type string
            df[message_col] = df[message_col].astype(str)
            
            # Créer les conditions pour np.select
            conditions = []
            choices = []
            
            # Définir les catégories spécifiques (les plus importantes en premier)
            specific_categories_order = [
                "Agression / Violence",
                "Harcèlement / Sexisme",
                "Malaise / Assistance",
                "Dégradation"
            ]

            for category in specific_categories_order:
                keywords = KEYWORDS_SECURITE[category]
                search_pattern = '|'.join(keywords)
                conditions.append(df[message_col].str.contains(search_pattern, case=False, na=False))
                choices.append(category)

            default_choice = 'Incivilité / Conflit / Autre'
            df['Sous_Categorie'] = np.select(conditions, choices, default=default_choice)
            
            # --- APPLICATION DE VOTRE RÈGLE ---
            # Si la colonne 'Nature_Clean' n'est pas 'sécurité',
            # alors la Sous_Categorie devient 'Non concerné'.
            # !! IMPORTANT : J'ai ajouté 'violence' et 'harcèlement' car vos données
            # utilisent ces termes dans la colonne 'Catégorie'
            natures_securite = ['sécurité', 'violence physique', 'violence verbale', 'harcèlement sexiste', 'violence sexuelle']
            df['Sous_Categorie'] = df['Sous_Categorie'].where(df['Nature_Clean'].str.lower().isin(natures_securite), 'Non concerné')
        
        # --- Traitement des dates/heures ---
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True) 

        if heure_col:
            # Gère les formats HH:MM:SS et HH:MM
            time_series = pd.to_datetime(df[heure_col], format='%H:%M:%S', errors='coerce').dt.time
            # Si la conversion échoue (NaT), essayez HH:MM
            if time_series.isnull().all():
                 time_series = pd.to_datetime(df[heure_col], format='%H:%M', errors='coerce').dt.time
            
            df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + time_series.astype(str), errors='coerce')
            df['Heure_Jour'] = df['DateTime'].dt.hour
        else:
            st.warning("La colonne 'Heure' n'a pas été trouvée. Utilisation de la date seule.")
            df['DateTime'] = df[date_col]
            df['Heure_Jour'] = 0
            
        df.dropna(subset=['DateTime'], inplace=True)
        
        if df.empty:
            st.error("Aucune donnée valide n'a pu être chargée après le traitement des dates.")
            return None

        # --- Feature Engineering Temporelle ---
        df['Jour_Semaine_Num'] = df['DateTime'].dt.dayofweek # Lundi=0, Dimanche=6
        df['Jour_Semaine_Nom'] = df['DateTime'].dt.day_name() 
        df['Date_Seule'] = df['DateTime'].dt.date
        
        # Renomme la colonne nettoyée en 'Nature' pour le reste du script
        df.rename(columns={'Nature_Clean': 'Nature', 'Perimetre_Clean': 'Périmètre'}, inplace=True)

        return df
    
    except FileNotFoundError:
        st.error(f"Erreur critique : Le fichier '{file_path}' n'a pas été trouvé.")
        return None
    except Exception as e:
        with open("erreur_log.txt", "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("ERREUR INATTENDUE DANS LOAD_DATA (Code Version Corrigée):\n")
            f.write(f"Exception: {e}\n\n")
            f.write(traceback.format_exc()) # Affiche la trace complète
            f.write("\n" + "="*50 + "\n")
        
        st.error(f"Erreur lors du chargement ou du traitement du fichier : {e}. Vérifiez 'erreur_log.txt'.")
        return None

# --- Interface Principale ---
st.title("🚇 Dashboard d'Analyse des Signalements (Périmètre IA)")
st.info("Veuillez téléverser votre fichier CSV de signalements pour commencer.")

# --- MODIFICATION: Ajout du File Uploader ---
uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

# --- MODIFICATION: Ajout du sélecteur pour skiprows ---
rows_to_skip = st.number_input(
    "Lignes à ignorer en début de fichier (En-tête)", 
    min_value=0, 
    value=3, # Bon défaut pour "Signalements-.xlsx - Base .csv"
    help="Ajustez ce nombre pour ignorer les lignes non pertinentes en haut de votre CSV."
)

df_raw = None # Initialiser le DataFrame

# Continuer seulement si un fichier est téléversé
if uploaded_file is not None:
    # --- MODIFICATION: Appel de load_data avec les nouveaux paramètres ---
    df_raw = load_data(uploaded_file, rows_to_skip) 
else:
    st.warning("En attente d'un fichier CSV...")
    st.stop() # Arrêter l'exécution si aucun fichier n'est chargé

# --- Si les données ne sont pas chargées, arrêter l'app ---
if df_raw is None:
    st.error("Erreur critique lors du chargement des données. L'application ne peut pas démarrer.")
    st.warning("Vérifiez le nombre de lignes à ignorer ou les noms de colonnes dans votre CSV (ex: 'Date', 'Heure', 'Catégorie', 'Périmètre') ou regardez 'erreur_log.txt' pour les détails.")
    st.stop()
else:
    # --- Titre Principal de l'Application ---
    # st.title("🚇 Dashboard d'Analyse des Signalements (Périmètre IA)") # Déjà mis en haut

    min_date = df_raw['DateTime'].min().date()
    max_date = df_raw['DateTime'].max().date()

    with st.expander("Cliquez pour sélectionner la plage de dates 🗓️", expanded=True):
        
        # Gérer le cas où min_date et max_date sont identiques
        if min_date == max_date:
            default_range = (min_date, max_date)
        else:
            default_range = (min_date, max_date)

        date_range = st.date_input(
            "Sélectionnez une plage de dates",
            default_range,
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed" 
        )

    if len(date_range) == 2:
        date_debut = pd.to_datetime(date_range[0])
        date_fin = pd.to_datetime(date_range[1])
    else:
        # Gère le cas où une seule date est sélectionnée
        date_debut = pd.to_datetime(date_range[0])
        date_fin = pd.to_datetime(date_range[0])

    df_filtered = df_raw[
        (df_raw['DateTime'] >= date_debut) &
        (df_raw['DateTime'] <= (date_fin + pd.Timedelta(days=1))) # Inclure la journée de fin
    ]


    # --- Métriques Clés (KPIs) ---
    st.header("Statistiques Clés (selon dates sélectionnées)")

    if df_filtered.empty:
        st.warning("Aucune donnée ne correspond aux dates sélectionnées.")
    else:
        kpi1, kpi2, kpi3 = st.columns(3)
        
        kpi1.metric(
            label="Total Signalements Filtrés",
            value=f"{len(df_filtered):,}".replace(',', ' ')
        )
        
        kpi2.metric( 
            label="Périmètre Principal",
            value=df_filtered['Périmètre'].mode()[0]
        )
        
        kpi3.metric( 
            label="Nature Principale",
            value=df_filtered['Nature'].mode()[0]
        )

    st.divider()

    # --- Création des Onglets ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Aperçu des Données", 
        "📈 Analyse Sécurité", 
        "🕒 Analyse Temporelle"
    ])

    # --- Contenu de l'Onglet 1 : Aperçu & Données ---
    with tab1:
        st.header("Aperçu des Données (selon dates)")
        
        st.markdown(f"Affichage des **{len(df_filtered)}** signalements (selon les dates sélectionnées).")
        st.info("La colonne 'Sous_Categorie' est générée automatiquement par le script.")
        
        # Le dataframe affiche maintenant la nouvelle colonne 'Sous_Categorie'
        st.dataframe(df_filtered, use_container_width=True)
        
        st.markdown("### Informations sur les colonnes (Données Brutes)")
        with st.expander("Cliquez pour voir les détails des colonnes (types et valeurs nulles)"):
            buffer = StringIO()
            df_raw.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    # --- Contenu de l'Onglet 2 : Analyse Sécurité ---
    with tab2:
        st.header("Analyse par Nature et Périmètre")
        
        if df_filtered.empty:
            st.warning("Pas de données à afficher pour cette analyse.")
        else:
            
            sub_tab1, sub_tab2 = st.tabs(["Vue d'ensemble", "Détail Sécurité"])

            with sub_tab1:
                st.markdown("### Vue d'ensemble (Périmètre & Nature)")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top 10 Natures (Tous Périmètres)")
                    nature_counts = df_filtered['Nature'].value_counts().nlargest(10).reset_index()
                    nature_counts.columns = ['Nature', 'Nombre']
                    
                    fig_nature = px.bar(
                        nature_counts,
                        x='Nombre',
                        y='Nature',
                        orientation='h',
                        title="Top 10 Signalements (Tous Périmètres)",
                        color_discrete_sequence=[RATP_GREEN]
                    )
                    fig_nature.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        title_font_size=20, # Police réduite pour les graphiques
                        font_size=12         # Police réduite pour les graphiques
                    )
                    st.plotly_chart(fig_nature, use_container_width=True)
    
                with col2:
                    st.subheader("Répartition par Périmètre (Global)")
                    
                    perimetre_filtered_data = df_filtered[df_filtered['Périmètre'] != 'Non défini']
                    
                    if perimetre_filtered_data.empty:
                        st.warning("Aucune donnée de périmètre définie à afficher.")
                    else:
                        perimetre_counts = perimetre_filtered_data['Périmètre'].value_counts().reset_index()
                        perimetre_counts.columns = ['Périmètre', 'Nombre']
                        
                        perimetre_counts['Périmètre'] = perimetre_counts['Périmètre'].str.title()
                        
                        fig_perimetre = px.pie(
                            perimetre_counts,
                            names='Périmètre',
                            values='Nombre',
                            title="Proportion des Signalements par Périmètre",
                            color_discrete_sequence=px.colors.sequential.Greens_r
                        )
                        fig_perimetre.update_traces(textposition='inside', textinfo='percent+label')
                        fig_perimetre.update_layout(
                            title_font_size=20, # Police réduite
                            font_size=12,       # Police réduite
                            legend_title_text='Périmètre'
                        )
                        st.plotly_chart(fig_perimetre, use_container_width=True)
    
                st.divider()

                # --- AJOUT DE LA HEATMAP ---
                st.subheader("Croisement Nature / Périmètre (Top 10 Natures)")
                
                top_10_natures = df_filtered['Nature'].value_counts().nlargest(10).index
                
                df_heatmap_filtered = df_filtered[
                    (df_filtered['Périmètre'] != 'Non défini') &
                    (df_filtered['Nature'].isin(top_10_natures))
                ]
                
                if df_heatmap_filtered.empty:
                    st.warning("Pas de données croisées à afficher (Top 10 Natures vs. Périmètres définis).")
                else:
                    df_heatmap_counts = df_heatmap_filtered.groupby(['Nature', 'Périmètre']).size().reset_index(name='Nombre')
                    
                    fig_heatmap = px.density_heatmap(
                        df_heatmap_counts,
                        x='Périmètre',
                        y='Nature',
                        z='Nombre',
                        title="Heatmap des Signalements (Top 10 Natures vs. Périmètres)",
                        color_continuous_scale='Greens',
                        text_auto=True, 
                    )
                    
                    fig_heatmap.update_layout(
                        xaxis_title="Périmètre",
                        yaxis_title="Nature",
                        xaxis_tickangle=-45, 
                        yaxis={'categoryorder':'total descending'},
                        title_font_size=20, # Police réduite
                        font_size=12        # Police réduite
                    )
                    
                    # Affichage sur une seule colonne pour plus de largeur
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                # --- FIN DE L'AJOUT ---


            with sub_tab2:
                st.subheader("Détail des Signalements de Sécurité")
                st.markdown("Cette analyse lit les messages des signalements (uniquement pour la nature 'Sécurité' ou équivalents) et les classe automatiquement.")
                
                df_securite_sub = df_filtered[
                    (df_filtered['Sous_Categorie'] != 'Non concerné') &
                    (df_filtered['Sous_Categorie'] != 'N/A')
                ]
                
                if df_securite_sub.empty:
                    st.warning("Aucun signalement 'Sécurité' classifié trouvé pour cette période.")
                else:
                    st.info(f"Total de **{len(df_securite_sub)}** signalements 'Sécurité' classifiés (selon les dates).")
                    
                    sub_counts = df_securite_sub['Sous_Categorie'].value_counts().reset_index()
                    sub_counts.columns = ['Sous-Catégorie', 'Nombre']
                    
                    fig_sub_bar = px.bar(
                        sub_counts,
                        x='Nombre',
                        y='Sous-Catégorie',
                        orientation='h',
                        title="Nombre d'incidents par Sous-Catégorie 'Sécurité'",
                        color_discrete_sequence=[RATP_BLUE]
                    )
                    fig_sub_bar.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        title_font_size=20, # Police réduite
                        font_size=12        # Police réduite
                    )
                    
                    col1_sub, col2_sub = st.columns(2)
                    with col1_sub:
                        st.plotly_chart(fig_sub_bar, use_container_width=True)


    # --- Contenu de l'Onglet 3 : Analyse Temporelle ---
    with tab3:
        st.header("Analyse Temporelle des Signalements")
        
        if df_filtered.empty:
            st.warning("Pas de données à afficher pour cette analyse.")
        else:
            st.subheader("Évolution des Signalements par Jour")
            
            daily_counts = df_filtered.groupby('Date_Seule').size().reset_index(name='Nombre')
            daily_counts['Date_Seule'] = pd.to_datetime(daily_counts['Date_Seule']) 

            fig_line = px.line(
                daily_counts,
                x='Date_Seule',
                y='Nombre',
                title='Nombre de signalements par jour',
                markers=True
            )
            fig_line.update_traces(line_color=RATP_BLUE)
            fig_line.update_layout(
                title_font_size=20, # Police réduite
                font_size=12        # Police réduite
            )
            
            col1_line, col2_line = st.columns(2)
            with col1_line:
                st.plotly_chart(fig_line, use_container_width=True)
            
            st.divider()

            st.subheader("Signalements par Jour de la Semaine")
            
            weekly_counts = df_filtered.groupby(['Jour_Semaine_Num', 'Jour_Semaine_Nom']).size().reset_index(name='Nombre').sort_values('Jour_Semaine_Num')
            
            if not weekly_counts.empty:
                fig_weekly = px.bar(
                    weekly_counts,
                    x='Jour_Semaine_Nom',
                    y='Nombre',
                    title="Total des Signalements par Jour de la Semaine",
                    color_discrete_sequence=[RATP_GREEN]
                )
                fig_weekly.update_xaxes(categoryorder='array', categoryarray=weekly_counts['Jour_Semaine_Nom'])
                fig_weekly.update_layout(
                    title_font_size=20, # Police réduite
                    font_size=12        # Police réduite
                )
                
                col1_weekly, col2_weekly = st.columns(2)
                with col1_weekly:
                    st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.warning("Pas de données pour l'analyse par jour de la semaine.")

            st.divider()
            
            st.subheader("Signalements par Heure de la Journée")
            
            hourly_counts = df_filtered.groupby('Heure_Jour').size().reset_index(name='Nombre')
            
            if not hourly_counts.empty:
                fig_hourly = px.bar(
                    hourly_counts,
                    x='Heure_Jour',
                    y='Nombre',
                    title="Total des Signalements par Heure de la Journée",
                    color_discrete_sequence=[RATP_BLUE]
                )
                fig_hourly.update_xaxes(type='category', dtick=1)
                fig_hourly.update_layout(
                    title_font_size=20, # Police réduite
                    font_size=12        # Police réduite
                )
                
                col1_hourly, col2_hourly = st.columns(2)
                with col1_hourly:
                    st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.warning("Pas de données pour l'analyse par heure (colonne 'Heure' peut-être manquante ou mal formatée).")

