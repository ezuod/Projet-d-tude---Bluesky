import requests
import os
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from datetime import datetime, timezone
from psycopg2.extras import execute_batch
import psycopg2

load_dotenv()

# ── Bluesky ──────────────────────────────────────────────────────────────────
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "fake_news_project")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

BASE_URL = "https://bsky.social/xrpc"


# ── Connexion PostgreSQL ──────────────────────────────────────────────────────
def get_connection():
    """Retourne une connexion psycopg2 à partir des variables d'environnement."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def init_schema(conn):
    """Crée la table raw_posts si elle n'existe pas encore."""
    ddl = """
    CREATE TABLE IF NOT EXISTS raw_posts (
        id          SERIAL PRIMARY KEY,
        uri         TEXT UNIQUE NOT NULL,
        cid         TEXT NOT NULL,
        author      TEXT NOT NULL,
        text        TEXT,
        lang        TEXT,
        created_at  TIMESTAMPTZ,
        ingested_at TIMESTAMPTZ,
        like_count  INTEGER DEFAULT 0,
        repost_count INTEGER DEFAULT 0,
        reply_count  INTEGER DEFAULT 0,
        source      TEXT DEFAULT 'bluesky'
    );
    CREATE INDEX IF NOT EXISTS idx_raw_posts_lang ON raw_posts (lang);
    CREATE INDEX IF NOT EXISTS idx_raw_posts_ingested ON raw_posts (ingested_at);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print("✅ Schéma PostgreSQL vérifié / créé.")


# ── Bluesky ───────────────────────────────────────────────────────────────────
def create_session():
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        raise EnvironmentError(
            "Variables BLUESKY_HANDLE ou BLUESKY_PASSWORD manquantes dans .env"
        )
    url = f"{BASE_URL}/com.atproto.server.createSession"
    payload = {"identifier": BLUESKY_HANDLE, "password": BLUESKY_PASSWORD}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Erreur authentification Bluesky : {e}")


def search_posts(access_token, query="fake news", limit=50, max_pages=3):
    all_posts = []
    cursor = None
    for _ in range(max_pages):
        params = {"q": query, "limit": limit, "cursor": cursor}
        try:
            response = requests.get(
                f"{BASE_URL}/app.bsky.feed.searchPosts",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Erreur lors de la collecte (page ignorée) : {e}")
            break
        data = response.json()
        all_posts.extend(data.get("posts", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return all_posts


def filter_language(posts, allowed=("fr", "en")):
    filtered = []
    ingested_at = datetime.now(timezone.utc)
    for post in posts:
        text = post["record"].get("text", "")
        try:
            lang = detect(text)
            if lang in allowed:
                filtered.append({
                    "uri":          post["uri"],
                    "cid":          post["cid"],
                    "author":       post["author"]["handle"],
                    "text":         text,
                    "lang":         lang,
                    "created_at":   post["record"]["createdAt"],
                    "ingested_at":  ingested_at,
                    "like_count":   post.get("likeCount", 0),
                    "repost_count": post.get("repostCount", 0),
                    "reply_count":  post.get("replyCount", 0),
                    "source":       "bluesky",
                })
        except LangDetectException:
            continue
    return filtered


# ── Stockage PostgreSQL ───────────────────────────────────────────────────────
def store_posts(posts, conn):
    if not posts:
        print("ℹ️  Aucun post à insérer.")
        return

    query = """
        INSERT INTO raw_posts (
            uri, cid, author, text, lang,
            created_at, ingested_at,
            like_count, repost_count, reply_count,
            source
        )
        VALUES (
            %(uri)s, %(cid)s, %(author)s, %(text)s, %(lang)s,
            %(created_at)s, %(ingested_at)s,
            %(like_count)s, %(repost_count)s, %(reply_count)s,
            %(source)s
        )
        ON CONFLICT (uri) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            execute_batch(cur, query, posts)
        conn.commit()
        print(f"✅ {len(posts)} posts traités (doublons ignorés via ON CONFLICT)")
    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ Erreur lors de l'insertion en base : {e}")


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(access_token, conn):
    print("🚀 Démarrage de la collecte (PostgreSQL)...")

    #raw_posts = search_posts(access_token, query="fake news", limit=50, max_pages=3)
    """
    queries = [
    # FR — sujets sensibles
    "complot", "vérité cachée", "médias mentent",
    "vaccin dangereux", "traitement naturel cancer",
    # EN — sujets sensibles  
    "deep state", "they don't want you to know",
    "mainstream media lies", "vaccine injury",
    # Neutres — pour avoir un dataset équilibré
    "actualité", "breaking news"]
"""
    for query in ["iran","ormuze", "trump", "ukraine guerre", "xi jinping", "hantavirus"]:
        raw_posts = search_posts(access_token, query=query, limit=100, max_pages=5)
        clean_posts = filter_language(raw_posts)
        store_posts(clean_posts, conn)
    print(f"📥 {len(raw_posts)} posts bruts collectés")

    clean_posts = filter_language(raw_posts)
    print(f"🔍 {len(clean_posts)} posts après filtrage FR/EN")

    store_posts(clean_posts, conn)
    print("🏁 Pipeline terminé avec succès")


if __name__ == "__main__":
    session = create_session()
    token = session["accessJwt"]

    conn = get_connection()
    try:
        init_schema(conn)      # crée la table si besoin
        run_pipeline(token, conn)
    finally:
        conn.close()
