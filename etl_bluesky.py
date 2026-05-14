"""
etl_bluesky.py
--------------
ETL Data Engineer — pipeline Bluesky
Lit les posts bruts depuis raw_posts (PostgreSQL),
nettoie les textes, et stocke dans processed_posts.

Rôle : préparer les données pour le modèle NLP du Data Scientist.
"""

import os
import re
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

# ── Connexion PostgreSQL ──────────────────────────────────────────────────────
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB       = os.getenv("POSTGRES_DB", "fake_news_project")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")


def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# ── Création de la table processed_posts ─────────────────────────────────────
def init_schema(conn):
    """Crée la table processed_posts si elle n'existe pas encore."""
    ddl = """
    CREATE TABLE IF NOT EXISTS processed_posts (
        id              SERIAL PRIMARY KEY,
        raw_uri         TEXT UNIQUE NOT NULL REFERENCES raw_posts(uri),
        author          TEXT,
        lang            TEXT,
        original_text   TEXT,
        cleaned_text    TEXT,
        word_count      INTEGER,
        like_count      INTEGER DEFAULT 0,
        repost_count    INTEGER DEFAULT 0,
        reply_count     INTEGER DEFAULT 0,
        created_at      TIMESTAMPTZ,
        processed_at    TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_processed_lang
        ON processed_posts (lang);
    CREATE INDEX IF NOT EXISTS idx_processed_at
        ON processed_posts (processed_at);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print("✅ Table processed_posts vérifiée / créée.")


# ── Nettoyage du texte ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Nettoyage adapté aux posts courts de Bluesky.
    Cohérent avec la fonction clean_text du Data Scientist
    (preprocess.py Kaggle) mais adapté au format post/tweet.
    """
    if not isinstance(text, str):
        return ""

    # Suppression des URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"pic\.twitter\.com/\S+", "", text)
    text = re.sub(r"t\.co/\S+", "", text)

    # Suppression des mentions (@user)
    text = re.sub(r"@\w+", "", text)

    # Suppression des hashtags (garde le mot, retire le #)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Suppression des balises HTML résiduelles
    text = re.sub(r"<[^>]+>", "", text)

    # Suppression des emojis et caractères non-ASCII optionnel
    # (commenté : les emojis peuvent être utiles pour l'analyse émotionnelle)
    # text = text.encode("ascii", "ignore").decode("ascii")

    # Normalisation des espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Extraction depuis raw_posts ───────────────────────────────────────────────
def extract_raw_posts(conn) -> list[dict]:
    """
    Récupère les posts bruts pas encore traités
    (absents de processed_posts).
    """
    query = """
        SELECT
            r.uri,
            r.author,
            r.lang,
            r.text,
            r.like_count,
            r.repost_count,
            r.reply_count,
            r.created_at
        FROM raw_posts r
        LEFT JOIN processed_posts p ON r.uri = p.raw_uri
        WHERE p.raw_uri IS NULL
        ORDER BY r.ingested_at ASC;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    posts = [dict(zip(columns, row)) for row in rows]
    print(f"📥 {len(posts)} nouveaux posts bruts à traiter.")
    return posts


# ── Transformation ────────────────────────────────────────────────────────────
MIN_WORD_COUNT = 3  # filtre les posts trop courts après nettoyage

def transform(posts: list[dict]) -> tuple[list[dict], int]:
    """
    Applique le nettoyage et filtre les posts inutilisables.
    Retourne (posts_transformés, nb_filtrés).
    """
    transformed = []
    filtered = 0
    processed_at = datetime.now(timezone.utc)

    for post in posts:
        original_text = post["text"] or ""
        cleaned = clean_text(original_text)
        word_count = len(cleaned.split())

        # Filtre les posts vides ou trop courts après nettoyage
        if word_count < MIN_WORD_COUNT:
            filtered += 1
            continue

        transformed.append({
            "raw_uri":       post["uri"],
            "author":        post["author"],
            "lang":          post["lang"],
            "original_text": original_text,
            "cleaned_text":  cleaned,
            "word_count":    word_count,
            "like_count":    post["like_count"],
            "repost_count":  post["repost_count"],
            "reply_count":   post["reply_count"],
            "created_at":    post["created_at"],
            "processed_at":  processed_at,
        })

    print(f"🔍 {len(transformed)} posts valides après nettoyage ({filtered} filtrés trop courts).")
    return transformed, filtered


# ── Chargement dans processed_posts ──────────────────────────────────────────
def load(posts: list[dict], conn):
    """Insère les posts nettoyés dans processed_posts."""
    if not posts:
        print("ℹ️  Aucun post à insérer dans processed_posts.")
        return

    query = """
        INSERT INTO processed_posts (
            raw_uri, author, lang,
            original_text, cleaned_text, word_count,
            like_count, repost_count, reply_count,
            created_at, processed_at
        )
        VALUES (
            %(raw_uri)s, %(author)s, %(lang)s,
            %(original_text)s, %(cleaned_text)s, %(word_count)s,
            %(like_count)s, %(repost_count)s, %(reply_count)s,
            %(created_at)s, %(processed_at)s
        )
        ON CONFLICT (raw_uri) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            execute_batch(cur, query, posts)
        conn.commit()
        print(f"✅ {len(posts)} posts insérés dans processed_posts.")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Erreur lors de l'insertion : {e}")


# ── Rapport de synthèse ───────────────────────────────────────────────────────
def print_summary(conn):
    """Affiche un résumé de l'état de la base après ETL."""
    query = """
        SELECT
            lang,
            COUNT(*) AS total,
            ROUND(AVG(word_count), 1) AS avg_words,
            SUM(like_count) AS total_likes
        FROM processed_posts
        GROUP BY lang
        ORDER BY total DESC;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    print("\n📊 Résumé processed_posts :")
    print(f"{'Langue':<10} {'Posts':<10} {'Mots moy.':<12} {'Likes total'}")
    print("-" * 45)
    for row in rows:
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<12} {row[3]}")


# ── Pipeline principal ────────────────────────────────────────────────────────
def run_etl():
    print("🚀 Démarrage ETL Bluesky...")
    print(f"   Heure : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    conn = get_connection()
    try:
        init_schema(conn)

        # Extract
        raw_posts = extract_raw_posts(conn)
        if not raw_posts:
            print("ℹ️  Aucun nouveau post à traiter. ETL terminé.")
            return

        # Transform
        cleaned_posts, nb_filtered = transform(raw_posts)

        # Load
        load(cleaned_posts, conn)

        # Résumé
        print_summary(conn)

        print("\n🏁 ETL terminé avec succès.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_etl()
