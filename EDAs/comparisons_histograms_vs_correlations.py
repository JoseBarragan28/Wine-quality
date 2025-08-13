import streamlit as st        
import pandas as pd            
import matplotlib.pyplot as plt  


red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

col1, col2, col3 = st.columns([7, 7, 1.3])
with col3:
 x = st.selectbox("Eje x", list(red.columns))
 y = st.selectbox("Eje y", list(red.columns))
with col1:
 fig, (ax1,ax2) = plt.subplots(1,2)
 ax1.scatter(red[x], red[y], alpha=0.5)
 ax1.set_xlabel(x)
 ax1.set_ylabel(y)
 ax1.set_title(f'{x} vs {y} \n vino rojo')

 ax2.scatter(white[x], white[y], alpha=0.5)
 ax2.set_xlabel(x)
 ax2.set_ylabel(y)
 ax2.set_title(f'{x} vs {y}\n vino blanco')

 st.pyplot(fig)


with col2:
 #st.subheader("Correlación entre variables")
 correlation=red.corr()
 st.dataframe(correlation.style.background_gradient(cmap='coolwarm'),height=500)
 


