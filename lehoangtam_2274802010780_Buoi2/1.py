#bài 1
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import sys
sys.stdout.reconfigure(encoding='utf-8')
dataset = "F:\Học\Machineleaning\Machinelearning\Data\Education.csv" 
data = pd.read_csv(dataset,encoding="utf-8")
print("Dữ liệu đầu tiên:")
print(data.head())
label_encoder = LabelEncoder()
data['Label_2'] = label_encoder.fit_transform(data['Label']) 
print("\nDữ liệu sau khi mã hóa nhãn:")
print(data[['Text', 'Label', 'Label_2']].head())
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['Text']) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Huấn luyện mô hình Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
y_pred_bernoulli = bernoulli_nb.predict(X_test)
accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli)

# Huấn luyện mô hình Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test)
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)

# In kết quả
print(f"Accuracy Bernoulli Naive Bayes: {accuracy_bernoulli:.2f}")
print(f"Accuracy Multinomial Naive Bayes: {accuracy_multinomial:.2f}")



