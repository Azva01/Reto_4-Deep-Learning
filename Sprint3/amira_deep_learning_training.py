#Guion para limpiar con spacy, despues CountVectorizer,
# despues usar onehot encoding y entrenar modelo de Deep Learning con tf
# %%
# Importar bibliotecas
import os
import numpy as np
import spacy
import pandas as pd
import random
import seaborn as sns
import json
import pickle

import warnings
warnings.filterwarnings('ignore')

# import pyplot
from matplotlib import pyplot as plt

# Dummy Data from sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# tensorflow 
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout

from pprint import pprint

# %%
# spacy
try: 
    nlp = spacy.load('es_core_news_sm')
except OSError:
    print('Error: El modelo de spacy "es_core_news_sm no se encontró"')

# %%
#Carga de archivo de datos
#Ajuste para detectar ruta completa del .json
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_json = os.path.join(directorio_actual, 'conversations.json')

try:
    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Archivo cargado correctamente desde la ruta: {ruta_json}')
except FileNotFoundError:
    print(f'Error: El archivo no se encontro en {ruta_json}')


#pprint(data)

# %%
#Generar diccionario conversations_category_answers.json antes de limpieza
category_answers = {}
for conversation in data:
    cat = conversation['tag']
    category_answers[cat] = conversation['responses']

#(Entregable1)
with open('conversations_category_answers.json', 'w', encoding='utf-8') as f:
    json.dump(category_answers, f, ensure_ascii=False, indent=4)

#pprint(category_answers)
# %%
#Limpieza de datos con spacy previo a usar Sklearn
import itertools #mover a librerias

# lista para extraer preguntas
questions = []

for script in data:

    question = script['patterns']
    questions.append(question)

documents = list(itertools.chain.from_iterable(questions))

pd.DataFrame(data).explode(['patterns'])

documents
# %%
#limpiar signos de puntuacion, lematizacion y mayusculas

question_clean = []

for doc in documents:
    tokens = nlp(doc)
    #Quita puntuacion y lematiza
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]

    #Convierte en minsuculas
    new_tokens = [t.lower() for t in new_tokens]

    # Unir tokens limpios en sting con espacios
    question_clean.append(' '.join(new_tokens))

print(question_clean) #string con esapacios de tokens limpias para CountVect...

# %%
#Leer archivo de script de conversacion y extender por contenido de patterns
df_conversation = pd.DataFrame(data).explode(
    ['patterns']
    ).reset_index(drop=True)

df_conversation[['patterns', 'tag']]

# %%
#Usar CountVectorizer de Sklearn para hacer bolsa de palabras(bow)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

#Crear bow con lista de documentos par analisis
X = vectorizer.fit_transform(question_clean)

bow = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

#Salvar transformador que have bow (Entregable2)
pickle.dump(
    vectorizer,
    open('conversations_vectorizer_bow.pkl', 'wb')
)

bow.sample(15)

# %%
# Bolsa de palabras de cada pregunta con su categoria
processed_data = pd.concat(
    [bow,
     df_conversation[['tag']]
     ], axis=1)

processed_data.info()

# %%
#Reordenar datos aleatoriamente en dataframe:
processed_data = processed_data.sample(
    frac=1,
    random_state=123
).reset_index(drop=True)


# %%
#codificación con one hot encoding
y_dummies = pd.get_dummies(processed_data['tag'], dtype=int)
# %%
conversations_categories = list(y_dummies.columns)
#(Entregable3)
with open('conversations_categories.pkl', 'wb') as f:
    pickle.dump(conversations_categories, f)

print('Archivo "conversations_categories.pkl" exportado exitosamente.')


# %%
#Entrenamiento de modelo con Tf
dim_x = len(processed_data._get_numeric_data().to_numpy()[0])

dim_y = len(pd.get_dummies(processed_data['tag'], dtype=int).to_numpy()[0])

model = Sequential([
    Dense(25, input_shape=(dim_x,), activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dropout(0.2),
    Dense(dim_y, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

#bow con limpieza
X_train = processed_data._get_numeric_data().to_numpy()
y_train = pd.get_dummies(processed_data['tag'], dtype=float).to_numpy()

hist = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=10,
    verbose=1
)

# %%
#Guardar modelo
model.save('amira_model_trained.keras')
print('Modelo de red neuronal guardado correctamente.')


# %%
#Predecir respuesta
new_message = "Mi signo es Tauro"

def text_pre_process(message: str):

    tokens = nlp(message)
    new_tokens = [t.orth_ for t in tokens if not  t.is_punct]
    new_tokens = [t.lower() for t in new_tokens]
    clean_message = ' '.join(new_tokens)

    return clean_message

text_pre_process(new_message)

# %%
def bow_representation(message: str)-> np.array:

    bow_message = vectorizer.transform(
        [message]
        ).toarray()

    return bow_message

bow_representation(text_pre_process(new_message))

# %%
model(bow_representation(text_pre_process(new_message))).numpy()

# %%
print(conversations_categories)

# %%
def get_prediction(
        bow_message: np.array
        )-> int:
    """
    Obtiene la prediccion de la categoria
    que corresponde al mensaje
    """
    prediction = model(bow_message).numpy()

    index = np.argmax(prediction)

    predicted_category = conversations_categories[index] 

    return predicted_category

# %%
#Llamamos la funcion
get_prediction(
    bow_representation(
        text_pre_process(new_message)
        )
        )
# %%
def get_answer(category: str)-> str:
    """
    Obtiene el mensaje de respuesta para una categoria
    """
    # Obtiene las respuestas de la categoria
    answers = category_answers[category]

    # Selecciona una respuesta al azar
    ans = random.choice(answers)

    return ans

# %%
#Llamamos la funcion
bot_answer = get_answer(get_prediction(
    bow_representation(
        text_pre_process(new_message)
    )
))

# %%
print("Usuario:", new_message)
print("ChatBot:", bot_answer)
# %%
#Transformar .json en diccionario de busqueda rapida
conversations_answers = {item['tag']: item['responses'] for item in data}

os.makedirs('data', exist_ok=True)
with open('data/conversations_answers.pkl', 'wb') as f:
    pickle.dump(conversations_answers, f)

print('Archivo "conversations_answers.pkl" guardado exitosamente en data/')