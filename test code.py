import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import string

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Text Preprocessing Function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Example advertisement text
advert_text = """By 2050, 40% of our sales mix will be electricity⚡

Becoming a world-class player in the energy transition means meeting the growing demand for energy and adapting to changing customer expectations. We are developing a portfolio of operations across the electricity value chain through massive investments in #solar and #wind power and in #energystorage.

Discover our portfolio with link in bio."""

# Preprocess the advertisement text
processed_text = preprocess_text(advert_text)
print("Processed Text:", processed_text)

# Create a CountVectorizer to convert text to a bag of words
vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{3,}\b')
X = vectorizer.fit_transform([' '.join(processed_text)])

# Convert the document-term matrix to a gensim corpus
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

# Create an LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=1, id2word=id2word, passes=15)

# Print the topics found
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)

# Define keywords related to renewables
renewable_keywords = {'renewable', 'solar', 'wind', 'hydro', 'geothermal', 'biomass', 'green energy'}

# Check if any renewable keywords are in the processed text
contains_renewables = any(word in processed_text for word in renewable_keywords)

print("Contains renewables-related keywords:", contains_renewables)
