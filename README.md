# processing-morphology

## Project Overview

This project aims to create a pluralization system for English words using two approaches: a **Rule-Based Pluralizer** and a **Long Short-Term Memory (LSTM)** neural network model. The rule-based approach applies well-established rules for forming plurals, while the LSTM model is trained on a large dataset of singular and plural word pairs, allowing it to learn pluralization patterns from examples.

The primary goal of this project is to compare the performance of these two approaches on various categories of words, including regular and irregular plurals, as well as other edge cases like words ending with "y", "f", and uncommon or low-frequency plural forms.

## Features

- **Rule-Based Pluralization**: This method applies predefined linguistic rules to form plurals based on word endings or known irregularities.
  
- **LSTM Model**: A Bidirectional LSTM is trained to predict the plural forms of English words based on patterns observed in a training dataset.
  
- **Performance Evaluation**: The project compares the accuracy of both methods (LSTM and Rule-Based) across different word categories, such as regular plurals, irregular plurals, and edge cases like words with specific suffixes.

