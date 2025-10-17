# Mini-RAG pour l'Analyse de CVs : L'Assistant Recruteur

Ce projet a été développé en une semaine dans le cadre du processus de recrutement pour l'**Intelligence Lab de l'ECE**. L'objectif est de créer un prototype fonctionnel de **RAG (Retrieval-Augmented Generation)** permettant d'interroger une base de CVs en langage naturel.

---

## Fonctionnement du Projet

Le système RAG suit un pipeline classique, avec des choix spécifiques pour optimiser le compromis entre performance et rapidité sur une machine locale (CPU).

1.  **Split & Overlap** : Les CVs au format PDF sont découpés en petits morceaux (chunks) avec chevauchement pour maintenir la cohérence contextuelle.
2.  **Embedding** : Chaque chunk est transformé en un vecteur numérique, dimension 384 avec `paraphrase-multilingual-MiniLM-L12-v2`.
3.  **FAISS** : Une base de données vectorielle est créée pour indexer ces vecteurs, permettant une recherche de similarité ultra-rapide basée sur la similarité cosinus.
4.  **LLM** : Le modèle de langage (Mistral ou Phi3) utilise les chunks les plus pertinents récupérés par FAISS pour générer une réponse claire et concise.

### Modèles Utilisés :
* **LLM** : `mistral` ou `phi3:mini` via **Ollama**.
* **Embedding** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (principalement).

---

### Optimisation (Compromis Rapidité/Pertinence)

Le tableau suivant résume les expérimentations clés pour trouver le meilleur compromis entre vitesse et précision sur une machine limitée (CPU).

| LLM | Chunk Size | Overlap | Pertinence | Temps (s) |
| :---: | :---: | :---: | :---: | :---: |
| `phi3:mini` | 1000 | 200 | $\times$ | 50  |
| `mistral` | 1000 | 200 | $\checkmark$ | 140  |
| `mistral` | 500 | 100 | $\times$ | 40  |
| `phi3:mini` | 500 | 100 | $\times$ | 30  |
| `mistral` | 700 | 200 | $\times$ | 110  |

---

## Contenu du Dépôt

Vous trouverez les fichiers suivants dans ce répertoire :

* **`demo.py`** : Fichier principal contenant la logique du RAG et l'interface utilisateur **Gradio** pour une démonstration visuelle.
* **`mini_rag.ipynb`** : Un notebook Jupyter pour l'expérimentation, le développement et les tests unitaires des différentes étapes du RAG.
* **`test_opti`** : Fichiers de test utilisés pour évaluer les différentes combinaisons de chunk size, overlap, et de modèles LLM.
* **`DocTechnique.pdf`** : Le document technique complet décrivant les motivations, le fonctionnement détaillé, les problèmes rencontrés, les tests et les perspectives d'amélioration.

---

## Pour Aller Plus Loin

Une des pistes explorées (avec une pipeline schématisée dans le document technique) était l'implémentation d'un **prétraitement des données** pour extraire un format **JSON** structuré des CVs avant la vectorisation. Bien que cette approche prenne plus de temps par CV (environ 2 min 30s), elle permettrait une précision et une fiabilité bien supérieures en production.

L'utilisation d'un environnement **Cloud** est également une perspective d'amélioration pour la rapidité et la possibilité d'utiliser des modèles plus performants.

---

## Requis pour exécuter

Pour faire tourner le projet localement :

* **Python** (avec les packages `langchain`, `gradio`, `pypdf`
* **Ollama** installé et les modèles `mistral` ou `phi3:mini` téléchargés.

[Lien du GitHub du projet : https://github.com/axelbrons/MiniRAG]
