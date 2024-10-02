# Import spaCy library
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load sentiment model
nlp_senti = spacy.load("en_core_web_sm")

# Read csv file
import pandas as pd
review_df = pd.read_csv("amazon_product_reviews.csv")
print(review_df.info())

# Drop all columns besides review.text and review.rating
review_data = review_df[["reviews.text", "reviews.rating"]]

# Remove all NaNs
review_data = review_data.dropna()
print(review_data.info())

# Convert data into string objects
review_data = review_data.astype(str)

# Replace all punctunation with whitespaces, remove leading/trailing whitespaces, make reviews lower case
import string
review_list = [review.translate(str.maketrans(string.punctuation, " "*len(string.punctuation))) for review in list(review_data["reviews.text"])]
review_list = [review.strip() for review in review_list]
review_list = [review.lower() for review in review_list]

# Convert list of reviews undergoing preprocessing to dataframe
review_pro = pd.DataFrame({"reviews" : review_list})

# Store reviews in new column with sentiment stop words removed
review_pro["reviews_lemma_senti"] = review_pro.reviews.apply(lambda review: " ".join(word.lemma_ for word in nlp_senti(review) if not word.is_stop))

# Sentiment function
nlp_senti.add_pipe("spacytextblob")

def senti(review):
    doc = nlp_senti(review)
    return doc._.blob.sentiment

# Testing on every 3500th review in data (total of 10 reviews out of 34262) 
# Print review, rating and sentiment score (polarity) for comparison
for i in range(0,10):
    print("\n", review_data["reviews.text"][3500*i])
    print("Rating =", review_data["reviews.rating"][3500*i])
    print(senti(review_pro["reviews_lemma_senti"][3500*i]))


# ------------------------------------------Semantic model for two chosen reviews------------------------------------------

# Load semantic model
nlp_semant = spacy.load('en_core_web_md')

# Store reviews in new column of review_pro with sentiment stop words removed
review_pro["reviews_lemma_semant"] = review_pro.reviews.apply(lambda review: " ".join(word.lemma_ for word in nlp_semant(review) if not word.is_stop))

# Semantic function
def semant(review1, review2):
    review1 = nlp_semant(review1)
    review2 = nlp_semant(review2)
    return review1.similarity(review2)

# Testing comparison on first and second review in data
print("\n", review_data["reviews.text"][0])
print("\n", review_data["reviews.text"][1])
print("\n", "Semantic score of this pair of reviews is:", semant(review_pro["reviews_lemma_semant"][0], review_pro["reviews_lemma_semant"][1]), "\n")
# "This product so far has not disappointed. My children love to use it and I like the ability to monitor control what content they see with ease."
# "great for beginner or experienced person. Bought as a gift and she loves it"
#  0.6798489926067136
#  The two reviews share a decent amount of similarity; the product is loved and ease of use is expressed in both. 
#  This similarity score is reasonable for these two reviews