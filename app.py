import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df = pd.read_csv("trial_pets.csv")
df_n = df.drop(['Unnamed: 1','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11'],axis = 'columns')
df['Disease'] = le.fit_transform(df.Diseases)
dis = pd.concat([df_n,df['Disease']],axis='columns')
dis_n = dis.drop(['Diseases'],axis=True)
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_count = v.fit_transform(df.symptoms)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_count,dis_n.Disease)
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorzer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(df_n.symptoms,df_n.Diseases)
symptoms=[]

st.title("Dog Health Predictor :dog:")

nav = st.sidebar.radio("navigation",["Home","Predict","About Developer"])
if nav=="Home":
    st.image("dog.jpg",width=500)
    if st.checkbox("Show Data"):
        st.table(df_n)
    if st.checkbox("Health care tips"):
        st.video("https://www.youtube.com/watch?v=Tn3lZE0rRBs")
    if st.checkbox("Adaption"):
        x = st.selectbox("City",['Bengaluru','Mysore'],index=0)
        if x=='Bengaluru':
            st.write("https://www.justdial.com/Bangalore/Dog-Adoption-Centres/nct-10168823")
        if x=='Mysore':
            st.write("https://www.justdial.com/Mysore/Animal-Welfare-Organisations/nct-10017958")    
              
if nav=='Predict':
    st.header("Predict Disease")
    s = st.selectbox("Select Symptoms",["itching , hair fall , fever","vomit , loose motion","yellow vomit with foam , lot of burps","pubic area red and swollen , less urine , yellow urine , blood in urine , licking pubic area","inactive , vomit , blood loose motion with mucus , dehydrated , severe fever , increase in 2 days""inactive , vomit , blood loose motion with mucus , dehydrated , severe fever , increase in 2 days","vomit with undigested food","red tongue , drooling , drink plenty of water ","slow and painful urination , crystals in urine , fatigue , vomiting","straining to urinate , incontinence , urinate more , blood in urine","dripping urine , irritation , redness","mucus , green or yellow pus , watery eye discharge ","red and watery eyes , sensitive to light , squinting , rubbing the eyes , discharge from eyes ","lethargy , dry gums and tongue","eyes and nose discharge , coughing , improper vaccination , fever","tremors , weakness , disoriented , depression , drooling , diarrhea , seizures","collapsing , muscle twitching , drooling , chomping , foaming at mouth , loss of consciousness"],index=0)
    symptoms.append(s)
    pred = clf.predict(symptoms)

    if st.button("Predict"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress.progress(i+1)

        st.success(f"Your predicted disease is {pred}")
    
if nav=='About Developer':
    st.balloons()
    st.header("H M Navneet :sunglasses:")
    st.image("developer.jpg",width=400)
    st.write("Student at MS Ramaiah Institute of Technology")
    st.write("BE in CSE(Cyber Security)")
    st.write("LinkedIn : https://www.linkedin.com/in/h-m-navneet-859830236")
    st.write("GitHub : https://github.com/H-M-Navneet")