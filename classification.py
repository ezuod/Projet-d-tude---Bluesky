import torch
import psycopg2
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import sys

# ── Forcer l'encodage UTF-8 ───────────────────────────────────────────────────
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Charger les variables d'environnement ─────────────────────────────────────
# Charger depuis le fichier .env à la racine du projet
import pathlib

# Chercher le .env en remontant les répertoires
current_dir = pathlib.Path(__file__).parent
root_dir = current_dir.parent  # Remonte à la racine du projet
dotenv_path = root_dir / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path, encoding='utf-8')
else:
    load_dotenv(encoding='utf-8')

# ── Chemin du modèle ──────────────────────────────────────────────────────────
# Chemin relatif qui fonctionne sur n'importe quel ordinateur
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "ML", "RoBERTa", "roberta_finetuned_v2")

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost").strip()
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "fake_news_project").strip()
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres").strip()
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "").strip()


# ── Device (GPU ou CPU) ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device utilisé : {device}")


# ── Connexion PostgreSQL ──────────────────────────────────────────────────────
def get_connection():
    """Retourne une connexion psycopg2."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        client_encoding='UTF8',
    )


def load_posts_from_db(conn, limit=None):
    """Récupère les posts non classifiés de la base de données."""
    query = "SELECT id, text FROM raw_posts WHERE lang = 'en' ORDER BY id DESC"
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn)
    print(f"📥 {len(df)} posts chargés depuis PostgreSQL")
    return df


def load_model_and_tokenizer():
    """Charge le modèle fine-tuné et le tokenizer."""
    print(f"⏳ Chargement du modèle depuis {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    print("✅ Modèle et tokenizer chargés")
    return model, tokenizer


class PostDataset(Dataset):
    """Dataset pour les posts."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }


def classify_posts(model, tokenizer, texts, batch_size=32):
    """Classifie les posts avec le modèle fine-tuné."""
    print(f"\n🚀 Démarrage de la classification ({len(texts)} posts)...")
    
    dataset = PostDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classification"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Softmax pour obtenir les probabilités
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            confidences.extend(probs.max(dim=-1).values.cpu().numpy())
    
    print(f"✅ Classification terminée")
    return predictions, confidences


def save_results(df, predictions, confidences, output_file="classification_results.csv"):
    """Sauvegarde les résultats en CSV."""
    df["prediction"] = predictions
    df["confidence"] = confidences
    
    # Mapper les labels
    label_map = {0: "Not Fake News", 1: "Fake News"}
    df["label"] = df["prediction"].map(label_map)
    
    df.to_csv(output_file, index=False)
    print(f"💾 Résultats sauvegardés : {output_file}")
    
    # Statistiques
    print("\n📊 Statistiques :")
    print(df["label"].value_counts())
    print(f"\n⏳ Confiance moyenne : {df['confidence'].mean():.4f}")


def main():
    """Pipeline de classification."""
    try:
        # Charger le modèle
        model, tokenizer = load_model_and_tokenizer()
        
        # Charger les données
        conn = get_connection()
        df = load_posts_from_db(conn, limit=None)  # limit=None pour tous les posts
        conn.close()
        
        if len(df) == 0:
            print("⚠️  Aucun post trouvé dans la base de données!")
            return
        
        # Classifier
        predictions, confidences = classify_posts(model, tokenizer, df["text"].tolist())
        
        # Sauvegarder
        save_results(df, predictions, confidences)
        
        print("\n🏁 Pipeline de classification terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
