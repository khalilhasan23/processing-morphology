import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

# Helper function to vectorize words
def word_to_vector(word, max_length, char_to_index):
    word += "<EOS>"  # Add end-of-word token
    vector = np.zeros((max_length, len(char_to_index)))
    for i, char in enumerate(word):
        if i < max_length and char in char_to_index:
            vector[i, char_to_index[char]] = 1
    return vector

# Helper function to decode vectors back to words
def vector_to_word(vector, char_to_index):
    index_to_char = {v: k for k, v in char_to_index.items()}
    word = ""
    for row in vector:
        if row.sum() == 0:
            break  # Stop at padding
        char_index = np.argmax(row)
        char = index_to_char[char_index]
        if char == "<EOS>":
            break  # Stop decoding at EOS
        word += char
    return word

# Rule-Based Pluralizer
def rule_based_pluralizer(word):
    if word.endswith(("s", "x", "z", "sh", "ch")):
        return word + "es"
    elif word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    elif word.endswith("f"):
        return word[:-1] + "ves"
    elif word.endswith("fe"):
        return word[:-2] + "ves"
    elif word in irregular_nouns:
        return irregular_nouns[word]
    else:
        return word + "s"

# Irregular nouns
irregular_nouns = {
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice"
}

# File path
file_path = "plural.txt"

# Load data
def load_data(file_path):
    singulars, plurals = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Skip lines with missing values
            singular, plural, tags = parts[0], parts[1], parts[2]
            if "PL" in tags:
                singulars.append(singular)
                plurals.append(plural)
    return singulars, plurals

# Train LSTM Model
def train_lstm(file_path):
    # Load data
    singulars, plurals = load_data(file_path)

    # Limit dataset for training efficiency
    singulars = singulars[:8000]
    plurals = plurals[:8000]

    print(f"Training on {len(singulars)} words")

    # Build character vocabulary
    chars = sorted(set("".join(singulars + plurals) + "<EOS>"))  # Include end-of-word token
    char_to_index = {char: idx for idx, char in enumerate(chars)}

    # Define maximum word length
    max_length = max(max(len(word) for word in singulars), max(len(word) for word in plurals)) + 1  # Add space for <EOS>

    # Vectorize the data
    X = np.array([word_to_vector(word, max_length, char_to_index) for word in singulars])
    y = np.array([word_to_vector(word, max_length, char_to_index) for word in plurals])

    # Build LSTM model
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(max_length, len(char_to_index))),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Dense(len(char_to_index), activation="softmax")  # Character probabilities
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(X, y, epochs=50, batch_size=64, validation_split=0.1)

    return model, char_to_index, max_length

# Train LSTM Model
lstm_model, char_to_index, max_length = train_lstm(file_path)

# Define test sets for different categories
test_sets = {
    "Regular": [("book", "books"), ("car", "cars"), ("dog", "dogs"), ("chair", "chairs")],
    "Irregular": [("man", "men"), ("mouse", "mice"), ("tooth", "teeth"), ("child", "children")],
    "Low-Frequency": [("octopus", "octopuses"), ("cactus", "cacti"), ("radius", "radii"), ("syllabus", "syllabi")],
    "High-Frequency": [("boy", "boys"), ("girl", "girls"), ("cat", "cats"), ("house", "houses")],
    "Suffix -y": [("baby", "babies"), ("city", "cities"), ("berry", "berries")],
    "Suffix -f": [("knife", "knives"), ("leaf", "leaves"), ("wolf", "wolves")],
    "Short Words": [("car", "cars"), ("bat", "bats"), ("rat", "rats")],
    "Long Words": [("university", "universities"), ("dictionary", "dictionaries"), ("laboratory", "laboratories")]
}

# Evaluate model on different categories
for category, words in test_sets.items():
    singulars = [pair[0] for pair in words]
    true_plurals = [pair[1] for pair in words]

    # Vectorize test words
    X_test = np.array([word_to_vector(word, max_length, char_to_index) for word in singulars])

    # LSTM Predictions
    y_pred = lstm_model.predict(X_test)
    lstm_predicted_plurals = [vector_to_word(pred, char_to_index).split('<')[0] for pred in y_pred]

    # Rule-Based Predictions
    rule_based_predictions = [rule_based_pluralizer(word) for word in singulars]

    # Calculate Accuracy
    lstm_accuracy = sum(1 for true, pred in zip(true_plurals, lstm_predicted_plurals) if true == pred) / len(true_plurals)
    rule_based_accuracy = sum(1 for true, pred in zip(true_plurals, rule_based_predictions) if true == pred) / len(true_plurals)

    # Print Results
    print(f"\n### {category} Words ###")
    print(f"LSTM Accuracy: {lstm_accuracy * 100:.2f}%")
    print(f"Rule-Based Accuracy: {rule_based_accuracy * 100:.2f}%")

    # Show incorrect predictions
    for s, lstm_p, rule_p, true_p in zip(singulars, lstm_predicted_plurals, rule_based_predictions, true_plurals):
        if lstm_p != true_p or rule_p != true_p:
            print(f"{s}: LSTM → {lstm_p}, Rule-Based → {rule_p}, True → {true_p}")
