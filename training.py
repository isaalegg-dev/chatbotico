import random
import json
import pickle
import numpy as np 

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import gradient_descent_v2

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

all_words = []
classes = []
documents = []
ignore_char = [',', '.', '-']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenizar palabras utilizadas en intents.json

        pattern_words = nltk.word_tokenize(pattern)
        all_words.extend(pattern_words)

        # crea un documento por clase de intent junto con las palabras de cada patron

        documents.append((pattern_words, intent['tag']))

        # almacena los tipos de intents en la variable class

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# el vocabulario pasa a su forma base
vocabulary = [lemmatizer.lemmatize(word) for word in all_words if word not in ignore_char] 

# elimina palabras repetidas y son ordenadas 
vocabulary = sorted(set(vocabulary))

# se crean los archivos words.pkl y classes.pkl con la informacion obtenida con el codigo de arriba
pickle.dump(vocabulary, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in vocabulary:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# creacion del modelo de aprendizaje automatico
# se crea una red neuronal Sequential

model = Sequential()

# se a;ade las capas tipo Dense con 128 neuronas y que tenga el mismo tama;o que el array con el que se va a entrenar
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))

# las capas tipo dropout se utilizan para evital el sobreajuste del modelo
model.add(Dropout(0.5))

# funcion de activacion softmax implica dos cosas:
# 1) Convertir los datos procesados en probabilidades.
# 2) Que la suma de las probabilidades tenga como resultado un 1.

model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer = gradient_descent_v2.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#Entrenando y guardando el chatbot

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotico.h5', hist)

