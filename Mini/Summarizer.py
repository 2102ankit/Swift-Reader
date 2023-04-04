 
# pip install beautifulsoup4
# pip install lxml
# pip install nltk
# nltk.download('popular')

import nltk
import bs4 as bs
import urllib.request
import re
import heapq

def getSummary(url, numOfSentences):
    scraped_data = urllib.request.urlopen(url)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article,'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    
    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # print(article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    
    sentence_list = nltk.sent_tokenize(article_text)

    # print(sentence_list)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    

    summary_sentences = heapq.nlargest(numOfSentences, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

    return summary

numOfSentences = 5

url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'

url2 = "https://www.ndtv.com/india-news/china-releases-new-names-for-11-places-in-latest-arunachal-push-3918324"

url3 = "https://www.onmanorama.com/news/kerala/2023/04/04/kozhikode-train-fire-incident-up-man-under-scanner-police-reach-noida.html"

summary = getSummary(url3, numOfSentences)


print("\n\nSummary:\n")
print(summary)
