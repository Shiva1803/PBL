import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud


#dataset path
DATASET_PATH = "/Users/shivanshtripathi/Documents/Coding/PBL/input/Final cleaned dataset.csv"
dataset = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print(dataset.head())

# Assuming columns: 'theme' and 'text'
dataset = dataset.rename(columns={'label': 'theme', 'tweet': 'text'})
dataset = dataset[['theme', 'text']]

#Preprocessing Function

emojis = {
    ':)': 'smile', ':(': 'sad', ':P': 'raspberry', ';)': 'wink', ':-D': 'smile'
}
stopwordlist = set([
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and', 'any',
    'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between',
    'both', 'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most',
    'my', 'myself', 'now', 'o', 'of', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', 'should', 'so',
    'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
    'with', 'won', 'y', 'you', 'your', 'yours', 'yourself', 'yourselves'
])

def preprocess(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()
    
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for text in textdata:
        text = str(text).lower()
        text = re.sub(urlPattern, ' URL', text)
        for emoji in emojis.keys():
            text = text.replace(emoji, "EMOJI" + emojis[emoji])
        text = re.sub(userPattern, ' USER', text)
        text = re.sub(alphaPattern, " ", text)
        text = re.sub(sequencePattern, seqReplacePattern, text)

        tweetwords = ''
        for word in text.split():
            if len(word) > 1 and word not in stopwordlist:
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')
        processedText.append(tweetwords.strip())
    return processedText




#Preprocess Data

processedtext = preprocess(dataset['text'])
themes = dataset['theme']

X_train, X_test, y_train, y_test = train_test_split(processedtext, themes, test_size=0.2, random_state=42)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
vectoriser.fit(X_train)

X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)

print("Vectorisation complete!")


4#Training Models

def model_Evaluate(model, model_name):
    y_pred = model.predict(X_test)
    print(f"\nEvaluation Report for {model_name}")
    print(classification_report(y_test, y_pred))

    cf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()




#TLogistic Regression
LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel, "LogisticRegression")

#wordclouds for themes
def generate_wordclouds(dataset):
    themes = dataset['theme'].unique()
    for theme in themes:
        text = " ".join(dataset[dataset['theme'] == theme]['text'].astype(str))
        wordcloud = WordCloud(
            width=800, height=600,
            background_color='white',
            colormap='viridis',
            max_words=200
        ).generate(text)

        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud for Theme: {theme}", fontsize=16)
        plt.show()

generate_wordclouds(dataset)

#Saving the Models

with open("thematic_vectorizer.pkl", "wb") as f:
    pickle.dump(vectoriser, f)

with open("thematic_model.pkl", "wb") as f:
    pickle.dump(LRmodel, f)

print("Thematic model + vectorizer saved successfully!")


# Prediction Function
def predict_theme(vectoriser, model, text_list):
    textdata = vectoriser.transform(preprocess(text_list))
    preds = model.predict(textdata)
    df = pd.DataFrame({"text": text_list, "theme": preds})
    return df

if __name__ == "__main__":
    sample_text = ["AI is transforming education", "The election results were surprising"]
    result = predict_theme(vectoriser, LRmodel, sample_text)
    print("\nSample Prediction:\n", result)
