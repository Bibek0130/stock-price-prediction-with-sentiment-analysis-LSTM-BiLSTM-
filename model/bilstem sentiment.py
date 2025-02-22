# scrape data from any news portasl

import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
#from keras.preprocessing.text import Tokenizer
from preprocessing.text import Tokenizer  # ✅ Correct
from tensorflow import pad_sequences
from tensorflow import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow import Sequential
from tensorflow import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
dataset = "nepse_news_dataset_30k.csv"
df = pd.read_csv(dataset)

# Drop missing values
df.dropna(inplace=True)

# Clean text function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

df['News'] = df['News'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['News'], df['Label'], test_size=0.2, random_state=42)

# Tokenization & Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100  # Adjust based on dataset
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Convert labels to categorical (if multi-class)
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

# Build BiLSTM Model
embedding_dim = 128

model = Sequential([
    Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Change to softmax for multi-class
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predictions
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
