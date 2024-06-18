import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pandas as pd

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load CSV into a Pandas DataFrame
df = pd.read_csv('/Users/simonjputtock/Desktop/eib/Shell_Ads.csv')

# Preview the DataFrame
print(df.head())

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

# Define keywords related to renewables
renewable_keywords = {'renewable', 'solar', 'wind', 'hydro', 'geothermal', 'biomass', 'green', 'low-carbon'}

total_ads = len(df['ad_creative_bodies'])
ads_with_renewables = 0

for advert_text in df['ad_creative_bodies']:
    # Preprocess the advertisement text
    advert_text = str(advert_text)
    if advert_text == "":
        continue
    
    processed_text = preprocess_text(advert_text)
    
    if not processed_text:
        print(f"Skipping ad due to empty or stopword-only text: {advert_text}")
        continue

    # Check if there are any renewable keywords
    found_keywords = [word for word in processed_text if word in renewable_keywords]
    contains_renewables = len(found_keywords) > 0
    print("Contains renewables-related keywords:", contains_renewables)
    if contains_renewables:
        print("Keywords found:", found_keywords)
        ads_with_renewables += 1

    # Skip the vectorizer and LDA model if processed_text is empty
    if processed_text:
        # Create a CountVectorizer to convert text to a bag of words
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{3,}\b')
        try:
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
        except ValueError as e:
            print(f"Skipping ad due to ValueError: {e}")

# Calculate the percentage of ads containing renewables-related keywords
percentage_with_renewables = (ads_with_renewables / total_ads) * 100
print(f"Total ads: {total_ads}")
print(f"Ads containing renewables-related keywords: {ads_with_renewables}")
print(f"Percentage of ads with renewables-related keywords: {percentage_with_renewables:.2f}%")
