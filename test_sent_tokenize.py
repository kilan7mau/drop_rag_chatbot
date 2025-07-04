import nltk
nltk.download('punkt')

text = "This is sentence one. And this is sentence two."
sentences = nltk.sent_tokenize(text)
print(sentences)
