from streamlit_player import st_player
import streamlit as st
import time
# pip install "gensim==3.8.1"
from gensim.summarization import summarize

# Embed a youtube video
st_player("https://youtu.be/CmSKVW1v0xM")

# Embed a music from SoundCloud
st_player("https://soundcloud.com/imaginedragons/demons")


st_player("https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav")


symbols = ['./Audio Files/Meeting 1.mp3','./Audio Files/Meeting 2.mp3', './Audio Files/Meeting 3.mp3', './Audio Files/Meeting 4.mp3']

track = st.selectbox('Choose a the Meeting Audio',symbols)

st.audio(track)
data_dir = './inference-data/'

ratiodata = st.text_input("Please Enter a Ratio you want summary by: ")
if st.button("Generate a Summarized Version of the Meeting"):
    time.sleep(2.4)
    #st.success("This is the Summarized text of the Meeting Audio Files xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  xxxxxxgeeeeeeeeeeeeeee eeeeeeeeeeeeeehjjjjjjjjjjjjjjjsdbjhvsdk vjbsdkvjbsdvkb skbdv")
    
    
    if track == "./Audio Files/Meeting 2.mp3":
        user_input = "NKE"
        time.sleep(1.4)
        try:
            with open(data_dir + user_input) as f:
                st.success(summarize(f.read(), ratio=float(ratiodata)))          
                #print()
                st.warning("Sentiment: Negative")
        except:
            st.text("Company Does not Exist")

    else:
        user_input = "AGEN"
        time.sleep(1.4)
        try:
            with open(data_dir + user_input) as f:
                st.success(summarize(f.read(), ratio=float(ratiodata)))          
                #print()
                st.success("Sentiment: Positive")
        except:
            st.text("Company Does not Exist")


#st.text('This is the Summarized text of the Meeting Audio Files')

# st.audio('CantinaBand60.wav')

# st.audio('08dc3eb1-7e26-456f-ba2e-beb54d95f7b8.mp3')


