from pathlib import Path
import ast
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="SoundLab", page_icon="🎧", layout="wide")

FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]
MOODS = {
    "📚 Study":     [0.4, 0.25, 0.4, 0.8, 90],
    "🏋️ Sport":     [0.7, 0.9, 0.7, 0.1, 140],
    "😢 Tristesse": [0.1, 0.25, 0.3, 0.7, 80],
    "🎉 Soirée":    [0.8, 0.85, 0.85, 0.1, 125],
    "😌 Chill":     [0.6, 0.35, 0.6, 0.6, 100],
}


def first_artist(x):
    try:
        return ast.literal_eval(x)[0]
    except Exception:
        return "Unknown"


@st.cache_resource
def prepare():
    df = pd.read_csv(Path(__file__).parent / "data" / "tracks.csv")

    # cleaning
    df = df.dropna(subset=["name"]).drop_duplicates(subset="id")
    df = df[df["tempo"] > 0]
    df["year"] = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
    df = df[df["year"].between(1950, 2020)]
    df["year"] = df["year"].astype(int)
    df["duration_min"] = df["duration_ms"] / 60000
    df = df.sample(100_000, random_state=42).reset_index(drop=True)
    df["main_artist"] = df["artists"].apply(first_artist)

    # ML: scaling + KMeans
    scaler = StandardScaler().fit(df[FEATURES])
    X = scaler.transform(df[FEATURES])
    km = KMeans(n_clusters=8, random_state=42, n_init=10).fit(X)
    df["cluster"] = km.labels_

    # mood = closest mood target
    targets = pd.DataFrame(MOODS, index=FEATURES).T
    T = scaler.transform(targets[FEATURES])
    dist = ((X[:, None, :] - T[None, :, :]) ** 2).sum(axis=2)
    df["mood"] = np.array(list(MOODS))[dist.argmin(axis=1)]
    return df, scaler, km


df, scaler, km = prepare()

st.title("🎧 SoundLab")
tab1, tab2 = st.tabs(["🎭 Mood Mixer", "📊 Insights BI"])

# ---------------- TAB 1: MOOD MIXER ----------------
with tab1:
    mood = st.selectbox("Quel mood ?", list(MOODS))
    energy = st.slider("Énergie", 0.0, 1.0, float(MOODS[mood][1]))
    n = st.slider("Nombre de titres", 5, 30, 15)

    target = MOODS[mood].copy()
    target[1] = energy
    t_scaled = scaler.transform(pd.DataFrame([target], columns=FEATURES))

    cluster_id = km.predict(t_scaled)[0]
    pool = df[(df["cluster"] == cluster_id) & (df["popularity"] >= 30)].copy()
    pool["dist"] = ((scaler.transform(pool[FEATURES]) - t_scaled) ** 2).sum(axis=1)
    playlist = pool.nsmallest(n, "dist")

    st.subheader("🎶 Playlist")
    st.dataframe(playlist[["name", "main_artist", "year", "popularity"]])

    avg = playlist[["valence", "energy", "danceability", "acousticness"]].mean()
    fig = px.line_polar(r=avg.values, theta=avg.index, line_close=True)
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: BI ----------------
with tab2:
    st.header("Which moods should the platform focus on?")

    ms = df.groupby("mood").agg(
        titres=("name", "count"),
        popularité=("popularity", "mean"),
    ).reset_index()
    ms["part_catalogue"] = ms.titres / ms.titres.sum()
    ms["gap"] = (ms.popularité / ms.popularité.mean()
                 - ms.part_catalogue / ms.part_catalogue.mean())
    best = ms.nlargest(1, "popularité").iloc[0]
    gap = ms.nlargest(1, "gap").iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titres analysés", f"{len(df):,}")
    c2.metric("Popularité moyenne", f"{df.popularity.mean():.1f}")
    c3.metric("Mood le plus populaire", best.mood)
    c4.metric("Mood à fort potentiel", gap.mood)

    a, b = st.columns(2)
    a.plotly_chart(px.bar(ms, x="mood", y="titres", title="Offre : titres par mood"),
                   use_container_width=True)
    b.plotly_chart(px.bar(ms, x="mood", y="popularité", title="Demande : popularité moyenne"),
                   use_container_width=True)

    by_year = df.groupby(["year", "mood"]).popularity.mean().reset_index()
    st.plotly_chart(px.line(by_year, x="year", y="popularity", color="mood",
                            title="Évolution de la popularité par mood"),
                    use_container_width=True)

    st.warning(f"🎯 **Decision to investigate**: **{gap.mood}** has high popularity "
               "relative to its share of the catalogue. Correlation ≠ causation.")

    m = st.selectbox("Top artistes pour le mood :", list(MOODS), key="top")
    top = (df[df.mood == m].groupby("main_artist")
           .agg(titres=("name", "count"), popularité=("popularity", "mean"))
           .query("titres >= 5").nlargest(10, "popularité"))
    st.dataframe(top)

    with st.expander("✅ Data quality"):
        st.write("Doublons id :", int(df.duplicated("id").sum()))
        st.write("Valeurs manquantes :", int(df[FEATURES + ["name"]].isna().sum().sum()))
        st.write("Popularité entre 0 et 100 :", bool(df.popularity.between(0, 100).all()))
        st.write("Sous-ensemble : 100 000 titres (random_state=42)")

    with st.expander("📖 Définitions"):
        st.markdown("""
- **Mood**: the closest mood target (distance on scaled valence, energy, danceability, acousticness, tempo).
- **Popularité moyenne**: mean of Spotify `popularity` (0-100), a proxy for demand, not real streams.
- **Mood à fort potentiel**: relative popularity minus relative catalogue share. Positive = demand above supply.
- **Cluster**: KMeans group (k=8) used by the Mood Mixer.
""")