import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


dataset = "F:/Học/Machineleaning/Machinelearning/Data/Education.csv"
data = pd.read_csv(dataset, encoding="utf-8")


st.write("### Dữ liệu đầu tiên:")
st.write(data.head())


label_encoder = LabelEncoder()
data['Label_2'] = label_encoder.fit_transform(data['Label'])
st.write("### Dữ liệu sau khi mã hóa nhãn:")
st.write(data[['Text', 'Label', 'Label_2']].head())


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Text'])
y = data['Label_2']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
y_pred_bernoulli = bernoulli_nb.predict(X_test)
accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli)


multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test)
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)


st.write("### Kết quả:")
st.write(f"Accuracy Bernoulli Naive Bayes: {accuracy_bernoulli:.2f}")
st.write(f"Accuracy Multinomial Naive Bayes: {accuracy_multinomial:.2f}")
