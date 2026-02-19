#Aplicacion para crear API REST

#Importar librerias
# %%
import numpy as np
import spacy
import random
import pickle

import os
import flask
from flask import Flask, render_template, request, jsonify
from keras.models import load_model


# %%
#Cargamos modelo de esp de spacy
nlp = spacy.load('es_core_news_sm')
conversations = []
#Archivos de procesamiento
#Diccionario respuestas bot por categoría
conversations_answers = pickle.load(
    open('data/conversations_answers.pkl', 'rb')
)
conversations_categories = pickle.load(
    open('data/conversations_categories.pkl', 'rb')
)
#Transformador para texto a nube de palabras
vectorizer_bow = pickle.load(
    open('data/conversations_vectorizer_bow.pkl', 'rb')
)
#Modelo pre entrenado congelado en formato keras
model = load_model('data/amira_model_trained.keras') #formato nuevo .keras

# %%
def text_pre_process(message: str):
    '''
    Procesar texto de mensaje entrante
    '''

    tokens = nlp(message)
    #lematizacion y quitar puntuación
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]
    #minusculas
    new_tokens = [t.lower() for t in new_tokens]
    # unir tokens con un espacio
    clean_message = ' '.join(new_tokens)

    return clean_message #mensaje procesado


def bow_representation(message: str) -> np.array:
    '''
    Obtiene representacion del mensaje en su codif de la nube de palabras
    '''
    return vectorizer_bow.transform([message]).toarray()


def get_prediction(bow_message: np.array) -> str:
    '''
    obtiene preciccion de la categoria que corresponde al mensaje
    '''
    prediction = model.predict(bow_message, verbose=0)
    index = np.argmax(prediction)
    predicted_category = conversations_categories[index]

    return predicted_category


def get_answer(category: str) -> str:
    """
    Obtiene mensaje de respuesta para categoria
    """
    # Obtiene las respuestas de la categoria
    answers = conversations_answers[category]

    # Selecciona una respuesta al azar
    ans = random.choice(answers)

    return ans


# Instancia el app
app = Flask(__name__, template_folder='templates')


# Ruta raiz
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.form.get('question'):
        raw_q = request.form.get("question")
        print(f"Pregunta recibida: {raw_q}") #prueba

        clean_q = text_pre_process(raw_q)
        bow_q = bow_representation(clean_q)
        prediction = get_prediction(bow_q)
        bot_answer = get_answer(prediction)

        #Crear textos de respuesta y pregunta
        question = 'Usuario: ' + raw_q
        answer = 'Amira: ' + bot_answer

        #Guardar textos de conversacion en lista
        conversations.append(question)
        conversations.append(answer)
    
    return render_template('index.html', chat=conversations)


#Inicializa app
if __name__ == '__main__':
    app.run(debug=True)

