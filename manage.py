import numpy as np
np.random.seed(0)
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



negative_files = glob.glob('Negative/*.txt')
positive_files = glob.glob('Positive/*.txt')


def clean_text(text):
    from re import sub
    text = sub('[^ةجحخهعغفقثصضشسيىبلاآتنمكوؤرإأزدءذئطظ]', ' ', text)
    text = sub(' +', ' ', text)
    text = sub('[آإأ]', 'ا', text)
    text = sub('ة', 'ه', text)

    return text

negative_texts = []
positive_texts = []

for file in positive_files:
    with open(file, 'r', encoding='utf-8') as file_to_read:
        try:
            text = file_to_read.read()
            text = clean_text(text)
            if text == "":
                continue
            print(text)
            positive_texts.append(text)
            print("-" * 10)
        except UnicodeDecodeError:
            continue

for file in negative_files:
    with open(file, 'r', encoding='utf-8') as file_to_read:
        try:
            text = file_to_read.read()
            text = clean_text(text)
            if text == "":
                continue
            print(text)
            negative_texts.append(text)
            print("-" * 10)
        except UnicodeDecodeError:
            continue

print('count positive texts : ',len(positive_texts))
print('count negative texts : ',len(negative_texts))

positive_labels = [1]*len(positive_texts)
negative_labels = [0]*len(negative_texts)

all_texts = positive_texts + negative_texts
all_labels = positive_labels + negative_labels

print(len(all_labels) == len(all_texts))



all_texts, all_labels = shuffle(all_texts, all_labels)

x_train, x_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.20)


vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

vectorizer.fit(x_train)

x_train = vectorizer.transform(x_train)


model = LinearSVC()
model.fit(x_train, y_train)


x_test = vectorizer.transform(x_test)

predictions = model.predict(x_test)
print("Acc svm")
print(accuracy_score(y_test, predictions))


model = MultinomialNB()
model.fit(x_train, y_train)


predictions = model.predict(x_test)
print("Acc naive_bayes")
print(accuracy_score(y_test, predictions))


import pickle

with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)


with open('vectorizer.pickle', 'wb') as file:
    pickle.dump(vectorizer, file)


with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)


example_test = 'أنا سعيد جدا، كانت الرحلة رائعة'

cleaned_example_test = clean_text(example_test)


example_test_vector = vectorizer.transform([cleaned_example_test])


example_result = model.predict(example_test_vector)
print(example_result[0])
