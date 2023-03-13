import streamlit as st
import pandas as pd
import numpy as np
import pickle

movies_dict= pickle.load(open('movie_dict.pkl','rb'))
movies= pd.DataFrame(movies_dict)

similarity= pickle.load(open('similarity.pkl','rb'))

def recommend(movie):
    movie_index= movies[movies['title']==movie].index[0]
    distances= similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True,key= lambda x:x[1])[1:7]
    
    recommended_movies=[]
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


st.title('Movie Recommender System')

option = st.selectbox(
    'movie list',
    (movies['title'].values))

if st.button('Recommend'):
    recommendation= recommend(option)
    for i in recommendation:
        st.write(i)
    
