"""
Script d'analyse émotionnelle des posts Bluesky
Auteur: Étudiant 2 (Data Scientist) - Projet Thumalien
Date: 2026
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import warnings
warnings.filterwarnings('ignore')

# ====================================
# 1️⃣ CONFIGURATION
# ====================================

# Configuration PostgreSQL
DB_CONFIG = {
    'host': 'localhost',        # Adapter selon ton setup
    'port': 5432,
    'database': 'bluesky_db',   # Nom de ta DB
    'user': 'ton_user',         # Ton user PostgreSQL
    'password': 'ton_password'  # Ton mot de passe
}

# Configuration du modèle
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
BATCH_SIZE = 32                 # Ajuster selon RAM/GPU
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration des fichiers de sortie
OUTPUT_DIR = "./emotion_analysis_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"{OUTPUT_DIR}/emotions_{TIMESTAMP}.csv"

# Labels des émotions
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# ====================================
# 2️⃣ CLASSE ANALYSEUR D'ÉMOTIONS
# ====================================

class EmotionAnalyzer:
    """Analyseur d'émotions optimisé pour batch processing"""
    
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        print(f"🎭 Chargement du modèle: {model_name}")
        print(f"🔧 Device: {device}")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.emotions = EMOTIONS
        
        print("✓ Modèle chargé avec succès\n")
    
    def predict_batch(self, texts, batch_size=BATCH_SIZE):
        """
        Prédire les émotions pour une liste de textes
        
        Args:
            texts: List[str] - Liste des textes à analyser
            batch_size: int - Taille des batchs
        
        Returns:
            List[dict] - Liste des prédictions avec scores
        """
        all_predictions = []
        
        # Traitement par batch avec barre de progression
        for i in tqdm(range(0, len(texts), batch_size), desc="🎭 Analyse émotions"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenization
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Conversion en dictionnaire
            for prob in probs.cpu().numpy():
                emotions_dict = {
                    f'emotion_{emotion}': float(score) 
                    for emotion, score in zip(self.emotions, prob)
                }
                
                # Ajouter l'émotion dominante
                dominant_idx = np.argmax(prob)
                emotions_dict['dominant_emotion'] = self.emotions[dominant_idx]
                emotions_dict['dominant_score'] = float(prob[dominant_idx])
                
                all_predictions.append(emotions_dict)
        
        return all_predictions

# ====================================
# 3️⃣ FONCTIONS UTILITAIRES
# ====================================

def create_output_directory():
    """Créer le dossier de sortie s'il n'existe pas"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Dossier de sortie: {OUTPUT_DIR}")

def connect_to_database():
    """Connexion à PostgreSQL"""
    try:
        print("🔌 Connexion à la base de données...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connecté à PostgreSQL\n")
        return conn
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        raise

def fetch_posts_from_db(conn, limit=None, lang_filter=None):
    """
    Récupérer les posts depuis PostgreSQL
    
    Args:
        conn: Connexion psycopg2
        limit: Nombre max de posts (None = tous)
        lang_filter: Filtrer par langue ('en', 'fr', etc.)
    
    Returns:
        DataFrame avec les posts
    """
    print("📥 Récupération des posts depuis la base...")
    
    # Construction de la requête SQL
    query = """
        SELECT 
            id,
            raw_uri,
            author,
            lang,
            original_text,
            cleaned_text,
            word_count,
            like_count,
            repost_count,
            reply_count,
            created_at,
            processed_at
        FROM processed_posts
        WHERE cleaned_text IS NOT NULL 
          AND cleaned_text != ''
    """
    
    # Ajouter filtre langue si spécifié
    if lang_filter:
        query += f" AND lang = '{lang_filter}'"
    
    # Ajouter ordre chronologique
    query += " ORDER BY created_at DESC"
    
    # Ajouter limite si spécifiée
    if limit:
        query += f" LIMIT {limit}"
    
    # Exécution
    try:
        df = pd.read_sql_query(query, conn)
        print(f"✓ {len(df)} posts récupérés")
        
        # Statistiques
        print(f"\n📊 Statistiques:")
        print(f"   - Langues: {df['lang'].value_counts().to_dict()}")
        print(f"   - Plage de dates: {df['created_at'].min()} → {df['created_at'].max()}")
        print(f"   - Mots moyen/post: {df['word_count'].mean():.1f}\n")
        
        return df
    
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        raise

def analyze_emotions(df, analyzer):
    """
    Analyser les émotions de tous les posts
    
    Args:
        df: DataFrame avec les posts
        analyzer: Instance de EmotionAnalyzer
    
    Returns:
        DataFrame enrichi avec les émotions
    """
    print("🎭 Début de l'analyse émotionnelle...")
    
    # Utiliser cleaned_text, sinon original_text
    texts = df['cleaned_text'].fillna(df['original_text']).tolist()
    
    # Analyse par batch
    predictions = analyzer.predict_batch(texts)
    
    # Convertir en DataFrame
    emotions_df = pd.DataFrame(predictions)
    
    # Combiner avec le DataFrame original
    result_df = pd.concat([df.reset_index(drop=True), emotions_df], axis=1)
    
    print("✓ Analyse terminée\n")
    return result_df

def save_to_csv(df, output_path):
    """Sauvegarder les résultats en CSV"""
    print(f"💾 Sauvegarde des résultats...")
    
    # Réorganiser les colonnes
    base_cols = [
        'id', 'raw_uri', 'author', 'lang', 'created_at',
        'original_text', 'cleaned_text', 'word_count',
        'like_count', 'repost_count', 'reply_count',
        'dominant_emotion', 'dominant_score'
    ]
    
    emotion_cols = [f'emotion_{e}' for e in EMOTIONS]
    
    all_cols = base_cols + emotion_cols + ['processed_at']
    
    # Filtrer colonnes existantes
    available_cols = [col for col in all_cols if col in df.columns]
    
    # Sauvegarder
    df[available_cols].to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Fichier sauvegardé: {output_path}")
    print(f"   - Nombre de lignes: {len(df)}")
    print(f"   - Nombre de colonnes: {len(available_cols)}\n")

def generate_statistics(df):
    """Générer des statistiques sur les émotions détectées"""
    print("="*60)
    print("📊 STATISTIQUES ÉMOTIONNELLES")
    print("="*60)
    
    # Distribution des émotions dominantes
    print("\n1️⃣ Distribution des émotions dominantes:")
    emotion_counts = df['dominant_emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {emotion:10s}: {count:5d} ({pct:5.1f}%)")
    
    # Score moyen par émotion
    print("\n2️⃣ Scores moyens par émotion:")
    for emotion in EMOTIONS:
        col_name = f'emotion_{emotion}'
        if col_name in df.columns:
            mean_score = df[col_name].mean()
            print(f"   {emotion:10s}: {mean_score:.3f}")
    
    # Confiance moyenne des prédictions
    print(f"\n3️⃣ Confiance moyenne: {df['dominant_score'].mean():.3f}")
    print(f"   Min: {df['dominant_score'].min():.3f}")
    print(f"   Max: {df['dominant_score'].max():.3f}")
    
    # Corrélation émotions vs engagement
    if 'like_count' in df.columns:
        print("\n4️⃣ Top 3 émotions par engagement (likes):")
        emotion_engagement = df.groupby('dominant_emotion')['like_count'].mean().sort_values(ascending=False).head(3)
        for emotion, avg_likes in emotion_engagement.items():
            print(f"   {emotion:10s}: {avg_likes:.1f} likes en moyenne")
    
    print("="*60 + "\n")

# ====================================
# 4️⃣ SCRIPT PRINCIPAL
# ====================================

def main():
    """Fonction principale"""
    
    print("="*60)
    print("🎭 ANALYSE ÉMOTIONNELLE - PROJET THUMALIEN")
    print("="*60 + "\n")
    
    # Étape 1: Préparation
    create_output_directory()
    
    # Étape 2: Connexion DB
    conn = connect_to_database()
    
    try:
        # Étape 3: Récupération des posts
        # MODIFIER ICI pour filtrer/limiter
        df_posts = fetch_posts_from_db(
            conn,
            limit=500,        # None = tous les posts, ou mettre un nombre (ex: 1000)
            lang_filter='en'   # 'en' pour anglais, 'fr' pour français, None pour tous
        )
        
        if len(df_posts) == 0:
            print("⚠️ Aucun post à analyser")
            return
        
        # Étape 4: Chargement du modèle
        analyzer = EmotionAnalyzer()
        
        # Étape 5: Analyse émotionnelle
        df_results = analyze_emotions(df_posts, analyzer)
        
        # Étape 6: Sauvegarde CSV
        save_to_csv(df_results, OUTPUT_CSV)
        
        # Étape 7: Statistiques
        generate_statistics(df_results)
        
        print("✅ Traitement terminé avec succès!")
        print(f"📄 Résultats disponibles dans: {OUTPUT_CSV}\n")
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Fermeture connexion
        if conn:
            conn.close()
            print("🔌 Connexion DB fermée")

# ====================================
# 5️⃣ POINT D'ENTRÉE
# ====================================

if __name__ == "__main__":
    main()