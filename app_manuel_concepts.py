import streamlit as st

"""
App manuel complémentaire : concepts avancés Paludisme & Rougeole
Ce module ajoute des explications pédagogiques sur les notions utilisées
(R₀, immunité collective, encodage circulaire, RollingMean4/Std4, Gradient Boosting,
métriques R² / MAE / RMSE et validation croisée).

À lancer avec :
    streamlit run app_manuel_concepts.py
"""

# Réutiliser le même style que dans app_manuel.py
st.markdown(
    """
<style>
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .benefit-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .alert-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Manuel – Concepts avancés")
st.markdown("*Complément pédagogique au manuel principal Paludisme & Rougeole*")
st.markdown("---")


tab_r0, tab_dates, tab_saison, tab_rolling, tab_gb, tab_metrics = st.tabs(
    [
        "R₀ & immunité collective",
        "Dates & détection précoce",
        "Saisonnalité (encodage circulaire)",
        "Moyennes & variabilité sur 4 semaines",
        "Gradient Boosting",
        "Métriques & validation croisée",
    ]
)


# 1. R0 et immunité collective
with tab_r0:
    st.markdown(
        """
        <div class="info-card">
            <h3>R₀ et seuil de 95&nbsp;% de vaccination (Rougeole)</h3>
            <p><b>R₀</b> = taux de reproduction de base : nombre moyen de personnes infectées par un cas dans une population totalement sensible.</p>
            <ul>
                <li>Pour la rougeole, la littérature situe généralement R₀ entre <b>12 et 18</b> dans des populations non immunisées.</li>
                <li>Le <b>seuil théorique d’immunité collective</b> est donné par ≈ <code>1 - 1/R₀</code>.</li>
                <li>Pour R₀ = 12, le seuil est ≈ 92&nbsp;% ; pour R₀ = 18, ≈ 94–95&nbsp;%.</li>
            </ul>
            <p>En pratique, les programmes de vaccination visent une <b>couverture &ge; 95&nbsp;%</b> des enfants pour casser les chaînes de transmission.</p>
            <p><i>Lecture pour la présentation :</i> la rougeole est l’une des maladies humaines les plus contagieuses connues ; avec un R₀ de 12–18, laisser plus de 5–10&nbsp;% d’enfants non vaccinés suffit pour alimenter des flambées.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 2. Dates : début d’éruption vs notification
with tab_dates:
    st.markdown(
        """
        <div class="info-card">
            <h3>Dates épidémiologiques : Date_Debut_Eruption vs Date_Notification</h3>
            <p><b>Date_Debut_Eruption</b> : moment où les symptômes caractéristiques de la rougeole apparaissent (fièvre, éruption). C’est à partir de là que le patient est réellement contagieux.</p>
            <p><b>Date_Notification</b> : moment où le cas est enregistré dans le système (fiche de notification remplie, saisie, validation). Cette date arrive souvent <b>7 à 14 jours plus tard</b>.</p>
            <h4>Pourquoi corriger ce délai ?</h4>
            <ul>
                <li>Si on se base uniquement sur la <b>notification</b>, on observe l’épidémie avec un décalage pouvant aller jusqu’à 2 semaines.</li>
                <li>En recodant les cas selon la <b>Date_Debut_Eruption</b>, on se rapproche du moment réel de transmission.</li>
                <li>La plateforme corrige ce décalage pour permettre une <b>détection plus précoce</b> des signaux épidémiques (J0 au lieu de J+14).</li>
            </ul>
            <p><i>Formulation orale :</i> "La date de début d’éruption correspond au moment où la personne est vraiment malade et contagieuse, alors que la notification arrive après les démarches administratives. Si on regarde seulement la notification, on voit l’épidémie en retard."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 3. Encodage circulaire de la semaine (sin/cos)
with tab_saison:
    st.markdown(
        """
        <div class="info-card">
            <h3>Encodage circulaire de la semaine (sin_week, cos_week)</h3>
            <p><b>Problème :</b> si on donne au modèle simplement le numéro de semaine (1, 2, ..., 52), il croit que :</p>
            <ul>
                <li>La semaine 52 et la semaine 1 sont très éloignées (52 vs 1),</li>
                <li>alors qu’en réalité elles sont <b>côte à côte</b> dans le calendrier (fin décembre – début janvier).</li>
            </ul>
            <h4>Solution : encodage circulaire</h4>
            <p>On projette la semaine sur un cercle à l’aide de fonctions sinusoïdales :</p>
            <ul>
                <li><code>sin_week = sin(2π × semaine / 52)</code></li>
                <li><code>cos_week = cos(2π × semaine / 52)</code></li>
            </ul>
            <p>Sur ce cercle, les semaines 52 et 1 sont presque au même endroit, ce qui respecte la nature <b>cyclique</b> de la saisonnalité.</p>
            <h4>Intérêt pour la rougeole au Sahel</h4>
            <ul>
                <li>La rougeole présente une <b>périodicité annuelle</b> (pics qui reviennent à la même période chaque année).</li>
                <li>L’encodage sin/cos permet au modèle de mieux apprendre ce rythme saisonnier.</li>
            </ul>
            <p><i>Formulation orale :</i> "On met la semaine sur un cercle au lieu d’une droite, pour que la semaine 1 et la semaine 52 soient voisines comme dans le calendrier, ce qui aide le modèle à capter la saisonnalité."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 4. Moyenne et variabilité sur 4 semaines
with tab_rolling:
    st.markdown(
        """
        <div class="info-card">
            <h3>Moyenne et variabilité sur 4 semaines (RollingMean4, RollingStd4)</h3>
            <h4>Définitions</h4>
            <ul>
                <li><b>RollingMean4</b> = moyenne des cas sur les 4 semaines précédentes.</li>
                <li><b>RollingStd4</b> = variabilité (écart-type) des cas sur les 4 dernières semaines.</li>
            </ul>
            <h4>Pourquoi choisir 4 semaines ?</h4>
            <ul>
                <li>La rougeole a une <b>incubation d’environ 7–14 jours</b> : une génération de cas toutes les 1–2 semaines.</li>
                <li>Sur 4 semaines, on couvre typiquement <b>2 à 3 générations</b> de transmission successives.</li>
                <li>Cette fenêtre permet donc de capturer les <b>chaînes de transmission encore actives</b> au moment de la prédiction.</li>
            </ul>
            <h4>Interprétation</h4>
            <ul>
                <li><b>RollingMean4 élevé</b> = tendance épidémique installée (niveau de fond élevé).</li>
                <li><b>RollingStd4 élevé</b> = situation instable, avec des pics et creux fréquents.</li>
                <li>Ces deux indicateurs aident à distinguer une simple fluctuation normale d’un signal épidémique inhabituel.</li>
            </ul>
            <p><i>Formulation orale :</i> "Quatre semaines correspondent à deux à trois générations de transmission. La moyenne donne la tendance, et l’écart-type indique à quel point la situation est stable ou instable."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 5. Gradient Boosting
with tab_gb:
    st.markdown(
        """
        <div class="info-card">
            <h3>Gradient Boosting : principe et paramètres</h3>
            <h4>Idée générale</h4>
            <p>Le Gradient Boosting ne se contente pas d’un seul modèle. Il construit une <b>série de petits arbres de décision faibles</b> :</p>
            <ul>
                <li>On commence par un premier arbre qui fait une prédiction approximative.</li>
                <li>On regarde les <b>erreurs</b> de cet arbre.</li>
                <li>On entraîne un deuxième arbre pour corriger ces erreurs.</li>
                <li>On répète ce processus plusieurs fois ; à la fin, la <b>somme de tous les petits arbres</b> donne le modèle final.</li>
            </ul>
            <h4>Paramètres utilisés dans la plateforme</h4>
            <ul>
                <li><b>n_estimators = 200</b> : environ 200 petits arbres successifs.</li>
                <li><b>learning_rate = 0.05</b> : pas d’apprentissage <b>petit</b>, ce qui stabilise l’apprentissage et évite de trop s’adapter au bruit.</li>
                <li><b>max_depth = 4</b> : arbres peu profonds, qui apprennent des règles simples plutôt que des règles trop complexes.</li>
            </ul>
            <p><b>Pourquoi ce choix&nbsp;?</b> C’est un compromis entre :</p>
            <ul>
                <li><b>Capacité à capturer des relations non linéaires</b> (par exemple interactions climat × démographie × historique des cas),</li>
                <li>et <b>robustesse</b> (limiter le sur-apprentissage et garder un comportement stable sur de nouvelles données).</li>
            </ul>
            <p><i>Formulation orale :</i> "On utilise une série de petits arbres, chacun corrige les erreurs du précédent. On avance par petites corrections avec un pas de 0,05 et des arbres peu profonds, ce qui donne un modèle puissant mais relativement stable."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 6. Métriques de performance et validation croisée
with tab_metrics:
    st.markdown(
        """
        <div class="info-card">
            <h3>Métriques de performance : R², MAE, RMSE, CV‑R² ± σ</h3>
            <h4>R² (coefficient de détermination)</h4>
            <p><b>R²</b> mesure la proportion de la variabilité des cas observés que le modèle explique (entre 0 et 1).</p>
            <ul>
                <li><b>R² = 1</b> : prédiction parfaite.</li>
                <li><b>R² ≈ 0</b> : le modèle n’explique pas mieux qu’une moyenne constante.</li>
            </ul>

            <h4>MAE (Mean Absolute Error)</h4>
            <p>Le <b>MAE</b> est l’erreur absolue moyenne entre les cas prédits et les cas observés.</p>
            <p><i>Lecture :</i> "en moyenne, on se trompe de X cas par semaine".</p>

            <h4>RMSE (Root Mean Squared Error)</h4>
            <p>Le <b>RMSE</b> est la racine carrée de la moyenne des erreurs au carré. Il pénalise plus fortement les <b>grosses erreurs</b> que le MAE.</p>

            <h4>Validation croisée en série temporelle (CV‑R² ± σ)</h4>
            <p>La plateforme utilise une <b>validation croisée adaptée aux séries temporelles</b> :</p>
            <ul>
                <li>Les données sont découpées en plusieurs segments chronologiques (TimeSeriesSplit).</li>
                <li>On entraîne sur le passé et on teste sur le futur pour chaque segment.</li>
                <li>On calcule le <b>R² moyen</b> (CV‑R²) et sa <b>variabilité</b> (σ).</li>
            </ul>
            <p><b>Interprétation dans la plateforme :</b></p>
            <ul>
                <li><b>R² test &gt; 0.85</b> et <b>CV‑R² &gt; 0.80</b> → modèle jugé <b>fiable</b> pour l’usage opérationnel.</li>
                <li><b>R² test &lt; 0.70</b> → signal d’alerte : vérifier la qualité des données, la complétude ou les variables explicatives.</li>
            </ul>
            <p><i>Formulation orale :</i> "R² nous dit quelle part de la variabilité des cas on explique, MAE et RMSE de combien de cas on se trompe, et la validation croisée vérifie que ces bonnes performances sont stables d’un morceau de la série à l’autre."</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
