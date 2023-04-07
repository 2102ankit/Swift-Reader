# !pip install transformers
import time

t1 = time.perf_counter()
# import transformers
from transformers import pipeline
import os
import bs4 as bs
import urllib.request
import re

print("Time for import:")
print(time.perf_counter() - t1)

## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
# summarizer = pipeline("summarization")
t1 = time.perf_counter()
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    framework="pt",
    num_beams=2,
)
print("Time for model setup:")
print(time.perf_counter() - t1)
## To use the t5-base model for summarization:
# summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
# !!!!! ABOVE CODE GIVES ERROR I COULD NOT RESEOLVE BETTER TO USE SSH MODEL


# print(time.perf_counter())
t1 = time.perf_counter()


def getText(url):
    scraped_data = urllib.request.urlopen(url)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, "lxml")

    paragraphs = parsed_article.find_all("p")

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    article_text = re.sub(r"\[[0-9]*\]", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)
    return article_text


def getSummary(url):
    t1 = time.perf_counter()
    text = getText(url)
    # print("Time for function:")
    # print(time.perf_counter() - t1)

    # Summarize
    t1 = time.perf_counter()
    summary_text = summarizer(text, do_sample=True, truncation=True)[0]["summary_text"]
    # print("Time for summary:")
    # print(time.perf_counter() - t1)
    return summary_text


def main():
    url = "https://simple.wikipedia.org/wiki/Artificial_intelligence"
    urls = [
        "https://simple.wikipedia.org/wiki/Artificial_intelligence",
        "https://simple.wikipedia.org/wiki/Commodore_Nutt",
        "https://simple.wikipedia.org/wiki/Economics",
        "https://simple.wikipedia.org/wiki/Health",
        "https://simple.wikipedia.org/wiki/Anatomy",
        "https://simple.wikipedia.org/wiki/Human_rights",
        "https://simple.wikipedia.org/wiki/Hinduism",
        "https://simple.wikipedia.org/wiki/Movie",
    ]

    for url in urls:
        summary = getSummary(url)

        print("\nSummary:\n")
        print(summary.strip())


print("Time for compilation:")
print(time.perf_counter() - t1)

if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    print("Time for execution of main:")
    print(time.perf_counter() - t1)
