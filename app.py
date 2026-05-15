import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Spotify Track Analyzer & Recommender")

# --- Cached Functions for Performance ---

@st.cache_data
def load_data():
    # Re-download dataset if not already present (for standalone script)
    import kagglehub
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    data = pd.read_csv(f"{path}/dataset.csv")
    return data

@st.cache_data
def preprocess_data(data_df):
    # Make a copy to avoid modifying original cached data
    df = data_df.copy()

    # Handle 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)

    # Handle nulls (filling with most frequent as done in notebook)
    for col in ['artists', 'album_name', 'track_name']:
        if df[col].isnull().any():
            most_frequent = df[col].mode()[0] # .mode() returns a Series, take first
            df[col].fillna(most_frequent, inplace=True)

    # Split artists string, ensuring each element is a string before splitting
    df['artists'] = df['artists'].apply(lambda x: ','.join(str(x).split(';')))

    # Convert explicit column
    df['explicit'] = df['explicit'].astype(int)

    # Convert track and album names to title case
    df['track_name'] = df['track_name'].astype(str).apply(lambda x: x.lower().title())
    df['album_name'] = df['album_name'].astype(str).apply(lambda x: x.lower().title())

    # Convert duration_ms to minutes, seconds, hours
    df['duration_sec'] = df['duration_ms'] / 1000
    df['minutes'] = (df['duration_sec'] // 60).astype(int)
    df['seconds'] = (df['duration_sec'] % 60).round().astype(int)
    df['hours'] = (df['minutes'] // 60).astype(int)
    df.loc[df['minutes'] > 60, 'minutes'] = df.loc[df['minutes'] > 60, 'minutes'] % 60
    df.drop(columns='duration_ms', inplace=True)

    # Create trackDesc
    df.loc[df['speechiness'] > 0.66, 'trackDesc'] = 'Entire Spoken'
    df.loc[(df['speechiness'] >= 0.33) & (df['speechiness'] <= 0.66), 'trackDesc'] = 'Maybe Music and Speech'
    df.loc[df['speechiness'] < 0.33, 'trackDesc'] = 'Music'

    # Create isHit
    df['isHit'] = (df['popularity'] > 70).astype(int)

    # Create interaction terms as used in the last classification model
    df['en_danc_inrct'] = df['energy'] * df['danceability']
    df['loud_danc_intrct'] = df['loudness'] * df['danceability']
    df['acs_inst_intrc'] = df['acousticness'] * df['instrumentalness']

    return df

@st.cache_resource
def train_kmeans(df):
    features_for_kmeans = ['popularity', 'energy', 'danceability', 'loudness', 'acousticness',
                           'instrumentalness', 'liveness', 'valence', 'tempo', 'mode',
                           'explicit', 'key', 'speechiness', 'time_signature', 'minutes', 'seconds', 'hours']
    X_kmeans = df[features_for_kmeans]
    scaler_kmeans = StandardScaler() # It's good practice to scale for KMeans
    X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=15, random_state=42)
    kmeans.fit(X_kmeans_scaled)
    return kmeans, scaler_kmeans, features_for_kmeans

@st.cache_resource
def train_random_forest(df, kmeans_model, kmeans_scaler, kmeans_features):
    # Predict KMeans clusters for the full dataset
    X_kmeans_full = df[kmeans_features]
    X_kmeans_full_scaled = kmeans_scaler.transform(X_kmeans_full)
    df['kmeans_cluster'] = kmeans_model.predict(X_kmeans_full_scaled)

    # Features for the RandomForest model, including the new interaction terms and kmeans_cluster
    features_for_rf = ['liveness', 'explicit', 'minutes', 'seconds', 'hours', 'key',
                       'speechiness', 'time_signature', 'kmeans_cluster',
                       'en_danc_inrct', 'loud_danc_intrct', 'acs_inst_intrc']
    X_rf = df[features_for_rf]
    y_rf = df['isHit']

    scaler_rf = StandardScaler()
    X_rf_scaled = scaler_rf.fit_transform(X_rf)

    rf_model = RandomForestClassifier(random_state=42) # Using default params for simplicity in app
    rf_model.fit(X_rf_scaled, y_rf)
    return rf_model, scaler_rf, features_for_rf

@st.cache_data
def get_recommender_features(df, kmeans_model, kmeans_scaler, kmeans_features):
    # Ensure 'kmeans_cluster' is in the df for recommendation features
    if 'kmeans_cluster' not in df.columns:
        X_kmeans_full = df[kmeans_features]
        X_kmeans_full_scaled = kmeans_scaler.transform(X_kmeans_full)
        df['kmeans_cluster'] = kmeans_model.predict(X_kmeans_full_scaled)

    # All numerical features that describe a song's audio characteristics for similarity
    recommender_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                            'time_signature', 'minutes', 'seconds', 'hours', 'explicit', 'popularity', 'kmeans_cluster',
                            'en_danc_inrct', 'loud_danc_intrct', 'acs_inst_intrc']

    # Drop any NaNs that might have been introduced during feature engineering if not handled earlier
    df_clean = df.dropna(subset=recommender_features).reset_index(drop=True)

    scaler_recommender = StandardScaler()
    scaled_features = scaler_recommender.fit_transform(df_clean[recommender_features])

    df_features = pd.DataFrame(scaled_features, columns=recommender_features, index=df_clean.index)

    return df_clean, df_features, scaler_recommender, recommender_features


# --- Main App Logic ---

st.title("Spotify Tracks Analysis & Recommendation System")
st.write("Explore Spotify track data, predict 'hit' songs, and get personalized recommendations!")

# Load and Preprocess Data
with st.spinner("Loading and preprocessing data..."):
    raw_data = load_data()
    my_data_processed = preprocess_data(raw_data)
st.success("Data Loaded and Preprocessed!")

# Train Models (cached)
kmeans_model, kmeans_scaler, kmeans_features = train_kmeans(my_data_processed)
rf_model, rf_scaler, rf_features = train_random_forest(my_data_processed, kmeans_model, kmeans_scaler, kmeans_features)

# Get recommender features
df_recommender, df_scaled_features, recommender_scaler, recommender_features = get_recommender_features(my_data_processed, kmeans_model, kmeans_scaler, kmeans_features)

# --- Sidebar for Navigation/Inputs ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Hit Song Prediction", "Song Recommender", "Feature Distributions"])

if page == "Data Overview":
    st.header("Dataset Overview")
    st.write("Here's a glimpse of the processed Spotify tracks dataset:")
    st.dataframe(my_data_processed.head())
    st.write(f"Total tracks: {len(my_data_processed)}")
    st.write("Column Information:")
    st.write(my_data_processed.info(buf=None))

elif page == "Feature Distributions":
    st.header("Audio Feature Distributions")
    st.write("Explore the distribution of key audio features.")

    feature_to_plot = st.selectbox(
        "Select a feature to visualize",
        ["popularity", "danceability", "energy", "loudness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "speechiness"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(my_data_processed[feature_to_plot], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Distribution of {feature_to_plot.title()}")
    ax.set_xlabel(feature_to_plot.title())
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif page == "Hit Song Prediction":
    st.header("Predict if a Song is a 'Hit'")
    st.write("Adjust the parameters below to see if our model predicts the song to be a 'Hit' (Popularity > 70).")

    # User inputs for prediction
    st.sidebar.subheader("Prediction Inputs")
    popularity = st.sidebar.slider("Popularity", 0, 100, 50)
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5, 0.01)
    loudness = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -10.0, 0.1)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
    instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.01, 0.001)
    liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
    valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5, 0.01)
    tempo = st.sidebar.slider("Tempo (BPM)", 50, 200, 120)
    speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
    explicit = st.sidebar.selectbox("Explicit Content", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    key = st.sidebar.selectbox("Key", list(range(12)))
    time_signature = st.sidebar.selectbox("Time Signature", [0, 1, 3, 4, 5], index=3)
    duration_minutes = st.sidebar.slider("Duration (Minutes)", 1, 10, 3)
    duration_seconds = st.sidebar.slider("Duration (Seconds)", 0, 59, 30)
    duration_hours = st.sidebar.slider("Duration (Hours)", 0, 1, 0)

    # Create a DataFrame for the single input
    input_data = pd.DataFrame([[liveness, explicit, duration_minutes, duration_seconds,
                                duration_hours, key, speechiness, time_signature,
                                # KMeans cluster is predicted below
                                0, # Placeholder for kmeans_cluster
                                energy * danceability, # en_danc_inrct
                                loudness * danceability, # loud_danc_intrct
                                acousticness * instrumentalness # acs_inst_intrc
                               ]],
                              columns=['liveness', 'explicit', 'minutes', 'seconds', 'hours', 'key',
                                       'speechiness', 'time_signature', 'kmeans_cluster',
                                       'en_danc_inrct', 'loud_danc_intrct', 'acs_inst_intrc'])

    # Predict KMeans cluster for the single input
    kmeans_input_features = pd.DataFrame([[
        popularity, energy, danceability, loudness, acousticness,
        instrumentalness, liveness, valence, tempo, 0, # mode, assuming a default like 0 if not given
        explicit, key, speechiness, time_signature, duration_minutes, duration_seconds, duration_hours
    ]], columns=kmeans_features)
    kmeans_input_scaled = kmeans_scaler.transform(kmeans_input_features)
    input_data['kmeans_cluster'] = kmeans_model.predict(kmeans_input_scaled)[0]


    # Scale the input data for the Random Forest model
    scaled_input = rf_scaler.transform(input_data[rf_features])

    # Make prediction
    prediction = rf_model.predict(scaled_input)
    prediction_proba = rf_model.predict_proba(scaled_input)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"The model predicts this song is a **HIT**! (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.info(f"The model predicts this song is **NOT a Hit**. (Probability: {prediction_proba[0][0]:.2f})")

    st.write("--- Model Evaluation ---")
    st.write("The Random Forest model achieved an F1-Score of 0.7506 on the test set with initial feature engineering.")
    st.write("After further feature engineering, the F1-Score was 0.75.")

elif page == "Song Recommender":
    st.header("Find Similar Songs")
    st.write("Enter a song title or choose a song from the dropdown to get recommendations based on audio features.")

    # Option 1: Search by song title
    search_query = st.text_input("Search for a song title (e.g., 'Blinding Lights')")
    if search_query:
        # Case-insensitive search, partial match
        matching_songs = df_recommender[df_recommender['track_name'].str.contains(search_query, case=False, na=False)]
        if not matching_songs.empty:
            selected_song_info = matching_songs.iloc[0] # Take the first match
            st.write(f"Found: **{selected_song_info['track_name']}** by **{selected_song_info['artists']}**")
            target_song_index = selected_song_info.name
        else:
            st.write("No song found with that title. Try refining your search or selecting from the dropdown.")
            target_song_index = None
    else:
        target_song_index = None

    # Option 2: Select from dropdown if no search query or no match
    if target_song_index is None:
        available_songs = df_recommender['track_name'] + " - " + df_recommender['artists']
        selected_song_display = st.selectbox("Or select a song from the list", [''] + list(available_songs.unique()))

        if selected_song_display:
            # Extract track_name and artists from the display string
            track_name, artists = selected_song_display.rsplit(' - ', 1)
            selected_song_info = df_recommender[(df_recommender['track_name'] == track_name) & (df_recommender['artists'].str.contains(artists))].iloc[0]
            target_song_index = selected_song_info.name

    if target_song_index is not None:
        num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

        # Get the feature vector for the target song
        target_features = df_scaled_features.loc[target_song_index].values.reshape(1, -1)

        # Calculate cosine similarity with all other songs
        similarities = cosine_similarity(target_features, df_scaled_features.values)
        similarity_scores = list(enumerate(similarities[0]))

        # Sort songs by similarity score in descending order
        sorted_similarities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        st.subheader(f"Top {num_recommendations} Recommendations for {df_recommender.loc[target_song_index, 'track_name']}:")
        recommendations_df = pd.DataFrame(columns=['Track Name', 'Artist(s)', 'Album Name', 'Genre', 'Popularity', 'Similarity Score'])

        count = 0
        for i, score in sorted_similarities:
            if i == target_song_index: # Skip the song itself
                continue
            if count >= num_recommendations:
                break

            track = df_recommender.loc[i]
            recommendations_df.loc[count] = [
                track['track_name'],
                track['artists'],
                track['album_name'],
                track['track_genre'],
                track['popularity'],
                f"{score:.4f}"
            ]
            count += 1
        st.dataframe(recommendations_df)
    else:
        st.write("Please select a song to get recommendations.")

st.sidebar.markdown("---")
st.sidebar.write("Project by Your Name")
