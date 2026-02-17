#Guion para procesamiento de datos del archivo "conversations.json"
# %%
#importación de librerías
import os
import numpy as np
import spacy
import pandas as pd
import seaborn as sns
import json
import pickle

import warnings
warnings.filterwarnings('ignore')


# %%
#Importar modelo entrenado de spacy
try:
    nlp = spacy.load('es_core_news_sm')
#Avisar si no se encontro
except OSError:
    print("Error: el modelo de spacy 'es_core_news_sm no se encontró")

# %%
#Cargar archivo de datos
def procesar_datos_spacy():
    with open('conversations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    #Crear listas vacías
    vocabulario = []
    tags = []
    
    documentos = [] #lista para bolsa de palabras


    for item in data:
        tag = item['tag'] #Asignamos tag a su variable
        if tag not in tags:
            tags.append(tag) #Agregamos tag a la lista

        for pattern in item['patterns']: #Entrar a la lista de patterns
            frase_procesada = nlp(pattern.lower())# minusculas y a spacy
            
            tokens = [token.lemma_ for token in frase_procesada 
                      if not token.is_punct
            ]
            vocabulario.extend(tokens)
            documentos.append((tokens, tag))

    # set para quitar duplicados, list para recuperar formato lista.
    # sorted para ordenar en orden alfabetico
    vocabulario = sorted(list(set(vocabulario)))
    tags = sorted(list(set(tags)))

    # Generar Bolsa de palabras con documentos
    #documetos contiene dupla(tokens,tag)
    bolsa_palabras = []
    for dupla_astral in documentos:
        lista_de_tokens = dupla_astral[0]

        fila_binarios = []

        for palabra in vocabulario:
            if palabra in lista_de_tokens:
                fila_binarios.append(1)
            else:
                fila_binarios.append(0)

    #agregar fila a bolsa
    bolsa_palabras.append(fila_binarios)

    #Generar dataframe de pandas
    df_bolsa = pd.DataFrame(bolsa_palabras, columns=vocabulario)
    df_bolsa.to_csv('bow_amira_patterns.csv', index=False)

    print('Archivo bow_amira_patterns.csv creado exitosamente')

    #Guardar em archivo .pk
    #vocabulario.pk y tags.pk
    with open('vocabulario.pk', 'wb') as f:
        pickle.dump(vocabulario, f)
    with open('tags.pk', 'wb') as f:
        pickle.dump(tags, f)
    print ('Archivo .pk creado exitosamente')

#vectorizer =CountVectorizer()
#X = vectorizer.fit_transform()

# %%
#Correr la funcion
if __name__ == "__main__":
    procesar_datos_spacy()
# %%