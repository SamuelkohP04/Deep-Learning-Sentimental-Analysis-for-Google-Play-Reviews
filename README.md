# Deep-Learning-for-Sentimental-Analysis-for-Google-Play
The project aims to leverage Long Short-Term Memory (LSTM) models to analyze and predict the sentiment of Google Play Store reviews. The primary objective is to estimate the number of stars associated with each review, which reflects the overall sentiment or satisfaction level expressed by the reviewers.

### Model Architecture
This is our model architecture for LSTM:

![image](https://github.com/SamuelkohP04/Deep-Learning-Sentimental-Analysis-for-Google-Play-Reviews/assets/105436607/7f2bbc9f-17a6-4e29-8b64-791e5753fb48)

### Exploration of Word Embeddings

A total of 50 dimensions was visualised using Tensorboard. Using 3 Principal Components (PCs) from Principal Component Analysis (PCA), we observe the vector points of each word in 3-dimensional space:

![image](https://github.com/SamuelkohP04/Deep-Learning-Sentimental-Analysis-for-Google-Play-Reviews/assets/105436607/111ee551-52d5-4f2b-864d-273cac20408b)

For example, we can observe the nearest 100 neighbours of the vector of the word "good".

![image](https://github.com/SamuelkohP04/Deep-Learning-Sentimental-Analysis-for-Google-Play-Reviews/assets/105436607/e773017d-0235-4f51-9595-adea65d178fd)

This project consists of
- Paper to research and document findings of modelling approaches, and justifications on hyperparameter decision-making. Provides insights into the methodologies, techniques, and rationale behind the decision-making process for selecting hyperparameters. It serves as a comprehensive overview of the project's methodology and results.
- Data collection script, using web scraping to gather reviews into a Comma Separated Values (CSV) format. (data_collection_script ipynb file)
- CSV file after implementing web scraping from Google Play.
- Modelling processes using LSTM, from scratch (ipynb file)
- Model evaluation on custom inputs of reviews (ipynb file)
