import re
from pathlib import Path
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK resources (first run only)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ------------------ File Reading ------------------
def read_file(path):
    return Path(path).read_text(encoding="utf-8")

text = read_file("sample_text.txt")

# ------------------ Sentence Tokenization ------------------

# Without NLTK (simple regex)
def sentence_tokenize_simple(text):
    pattern = r'(?<=[.!?])\s+'
    return [s.strip() for s in re.split(pattern, text) if s.strip()]

sentences_simple = sentence_tokenize_simple(text)

# With NLTK (recommended)
sentences = sent_tokenize(text)

# ------------------ Word Tokenization ------------------
tokenized_sentences = [word_tokenize(s) for s in sentences]

# ------------------ Word Count & Unique Words ------------------
alpha_tokens = [tok for sent in tokenized_sentences for tok in sent if tok.isalpha()]
total_alpha_words = len(alpha_tokens)
unique_words = sorted({w.lower() for w in alpha_tokens})

# ------------------ Lowercasing ------------------
lowered_tokens = [w.lower() for w in alpha_tokens]

# ------------------ Stopword Removal & Frequency ------------------
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in lowered_tokens if w not in stop_words]
freq = Counter(filtered_tokens)
top5_words = freq.most_common(5)

# ------------------ Reusable Analysis Function ------------------
def analyze_text_file(path, top_n=5):
    text = read_file(path)
    sentences = sent_tokenize(text)
    tokenized = [word_tokenize(s) for s in sentences]
    alpha = [t for sent in tokenized for t in sent if t.isalpha()]
    lowered = [w.lower() for w in alpha]
    filtered = [w for w in lowered if w not in stop_words]
    frequency = Counter(filtered)

    return {
        "sentences": len(sentences),
        "total_words": len(alpha),
        "unique_words": len(set(lowered)),
        "top_words": frequency.most_common(top_n)
    }

# ------------------ Example Output ------------------
results = analyze_text_file("sample_text.txt")

print("Total sentences:", results["sentences"])
print("Total alpha-only words:", results["total_words"])
print("Unique words:", results["unique_words"])
print("\nTop frequent words:")
for word, count in results["top_words"]:
    print(f"{word} : {count}")
