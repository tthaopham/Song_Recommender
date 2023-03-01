import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn import datasets # sklearn comes with some toy datasets to practise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
import yellowbrick
import config
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit_player import st_player


st.title("DJs - Clemence et Thao")
song_input = st.text_input(
        "Enter your favourite song 👇"
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled,
        # placeholder=st.session_state.placeholder,
    )

artist_input = st.text_input(
        "Who is the artist singing this 👇"
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled,
        # placeholder=st.session_state.placeholder,
    )

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))

original_data = pd.read_csv('../df_concat_20905.csv')
original_data.drop_duplicates(inplace=True, ignore_index=False)

data_num = original_data.drop(['Unnamed: 0','Unnamed: 0.1','id', 'duration_ms'], axis=1)
data_num.head()

scaler = StandardScaler()
scaler.fit(data_num)

data_scaled = scaler.transform(data_num)

data_scaled_df = pd.DataFrame(data_scaled, columns = data_num.columns)

with open("scaler_spotify.pickle", "wb") as f:
    pickle.dump(scaler,f)

k_final = 8
kmeans = KMeans(n_clusters=k_final, random_state=1234)
kmeans.fit(data_scaled_df)
labels = kmeans.predict(data_scaled_df)
original_data["cluster"] = labels

filename = "kmeans_spotify_" + str(k_final) + ".pickle"
with open(filename, "wb") as f:
    pickle.dump(kmeans,f)

results = sp.search(q=str(song_input + artist_input),limit=1,market="GB")
track_id = results['tracks']['items'][0]['id']
track_features = sp.audio_features(track_id)

df = pd.DataFrame(track_features)
df=df[["danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo"]]

with open("./scaler_spotify.pickle", "rb") as f:
    saved_scaler = pickle.load(f) 

scaled_user_input = saved_scaler.transform(df)
input_label = kmeans.predict(scaled_user_input)

recommended_id = original_data[original_data['cluster'] == int(input_label)].sample()['id']
recommended_id = str(recommended_id.iloc[0])

def play_song(recommended_id):
    return IFrame(src="https://open.spotify.com/embed/track/"+recommended_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )

def recommend_song(userInputTrackId,saved_scaler,kmeans,original_data):
    ## get the audio features of the userSong
    track_id=sp.audio_features(userInputTrackId)[0]
    ## only filter the features that has been used for clustering
    df = pd.DataFrame(track_features)
    df=df[["danceability","energy","loudness","speechiness","acousticness", "instrumentalness","liveness","valence","tempo"]]

    ##scaling the new song
    scaled_user_input = saved_scaler.transform(df)
    
    ## find the closest cluster to the userSong
    input_label = kmeans.predict(scaled_user_input)[0]
    
    ## return the track id from a random song within the closest cluster
    recommended_id = original_data[original_data['cluster'] == int(input_label)].sample()['id']
    recommended_id = str(recommended_id.iloc[0])

    return recommended_id

import streamlit.components.v1 as components
from IPython.display import IFrame

st.write("This is your lemon:")
components.iframe("https://open.spotify.com/embed/track/"+track_id+"?utm_source=generator")

st.write("This is your lemonade:")
components.iframe("https://open.spotify.com/embed/track/"+recommended_id+"?utm_source=generator")

selectbox_basic = st.selectbox( "Do you like your lemonade?", ["_", "Yes", "No"] )
st.write(f"You selected {selectbox_basic}")

if selectbox_basic == 'Yes':
    st.write("YAY! Hope you enjoy it!!!")
elif selectbox_basic == 'No':
    st.write("Oops, we are sorry to hear, let's try again!")
else:
    st.write(" ")


