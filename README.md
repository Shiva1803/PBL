# PBL Project: Sentiment analysis model for analying tweets on X
<p> This ML model uses <strong>Logistic regression</strong> to analyze tweets and predict whether it's sentiment is positive or negative. 
  Side by side, one can see the stats with BernoulliNB and RandomForest aswell.
The project was initially trained on <a href = "https://www.kaggle.com/datasets/kazanova/sentiment140">Sentiment140</a> dataset (see updates section) and has been deployed via Streamlit.
  It also uses Flask API for the backend.
</p>

# Updates:
After usign sentiment140, Scrapping and preprocessing of 250k+ tweets was done manually using text cleaning, normalization, and stopword removal to give it an edge over the sentiment140 dataset.
Right now, the project is being upgraded to not just classify the sentiment, but the themes aswell (Political, Health, Finance).

# Preview of the project
<p align="center">
  <img src="images/Preview.png" alt="Preview" style="max-width: 100%; height: auto;" />
</p>




![Last Commit](https://img.shields.io/github/last-commit/Shiva1803/PBL)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
