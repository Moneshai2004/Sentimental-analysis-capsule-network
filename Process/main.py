# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Conv1D, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# Download NLTK and spacy data
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# Precompiled stopwords for efficiency
stop_words = set(stopwords.words('english'))

# Text processing function
def text_processing(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)  # Combine punctuation removal
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = ' '.join(word for word in word_tokenize(text) if word not in stop_words)
    return text

# Load data from CSV
def load_data(input_csv):
    try:
        data = pd.read_csv(input_csv)
        if 'body' not in data.columns:
            raise ValueError("Error: 'body' column not found. Please check the structure of the comments.")
        return data
    except Exception as e:
        print(e)
        return None

# Prepare data for Capsule Network
def prepare_data(data):
    data['Sentiment'] = data['Sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
    X = data['Processed_Text'].values
    y = data['Sentiment'].values
    
    # Tokenization
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=100)  # Adjust maxlen as necessary
    
    return X_padded, y, tokenizer

# Capsule Network model
def create_capsule_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Embedding(input_dim=5000, output_dim=64, input_length=input_shape[0])(inputs)
    x = Conv1D(filters=256, kernel_size=9, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)  # Added activation for improved learning
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=1)))(x)
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    return model

# Function to predict sentiment and aspects
def predict_sentiment(model, input_text, tokenizer):
    processed_text = text_processing(input_text)
    # Convert the processed text into sequences
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=100)  # Ensure the sequence is the same length as training data

    # Predict sentiment
    prediction = model.predict(padded_seq)
    predicted_sentiment = "Positive" if prediction[0] > 0 else "Negative"  # Simplified prediction logic

    # Dummy aspect analysis
    aspects = {
        "story": {
            "sentiment": "Positive",
            "score": 0.90
        },
        "acting": {
            "sentiment": "Positive",
            "score": 0.95
        },
        "direction": {
            "sentiment": "Neutral",
            "score": 0.50
        }
    }
    
    return {
        "input_text": input_text,
        "predicted_sentiment": predicted_sentiment,
        "score": prediction[0],
        "aspects": aspects
    }

# Main function for the sentiment analysis process
def main():
    input_csv = 'comments.csv'  # Change this to the path of your comments CSV
    data = load_data(input_csv)

    if data is None:
        return
    
    # Process text data
    data['Processed_Text'] = data['body'].apply(text_processing)

    # Prepare data
    X, y, tokenizer = prepare_data(data)
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Error: No data available for training.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check shapes of training data
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Create and train Capsule Network
    input_shape = (X_train.shape[1],)  # Define input shape based on padded sequences
    model = create_capsule_network(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Model Loss: {loss}, Model Accuracy: {accuracy}')

    # Save model weights
    model.save_weights('capsule_network_weights.weights.h5')  # Updated filename
    print("Model weights saved.")

    # Predict sentiments for all comments
    results = []
    for index, row in data.iterrows():
        result = predict_sentiment(model, row['body'], tokenizer)
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_csv = input("Output CSV file name for opinion results: ")
    results_df.to_csv(output_csv, index=False)
    print(f"Comments and sentiments saved to {output_csv}")

if __name__ == "__main__":
    main()
