import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer

# from pycaret.nlp import create_model, evaluate_model

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(
    page_title="Natural Language Processing App",
    page_icon="👋",
)

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    else:
        return []  # Return an empty list if the input is not a string

# Function to create word cloud
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# Title and description
st.title("Natural Language Processing Tools")
st.sidebar.success("Select a tool above.")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform topic modeling to visualize the topics using BERTopic.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Select text column
    text_column = st.selectbox("Select the text column for preprocessing", df.columns)

    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)

    # Show preprocessed text
    st.write("Preprocessed Text:")
    st.write(df[['processed_text']].head())

    # Filter out rows with empty or NaN processed_text
    df = df.dropna(subset=['processed_text'])

    # Prepare text for BERTopic
    documents = df['processed_text'].apply(lambda x: ' '.join(x))

    # Create BERTopic model
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(documents)

    # Show topics
    #st.write("BERTopic Topics:")
    #st.write(topic_model.get_topics())

    # Visualize topics
    st.write("Topic Visualization:")
    st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

 # Search for topics
    search_word = st.text_input("Enter a word to find related topics:")
    if search_word:
        search_topics, search_probabilities = topic_model.find_topics(search_word)
        search_results = pd.DataFrame({
            'Topic': search_topics,
            'Probability': search_probabilities
        }).sort_values(by='Probability', ascending=False)
        st.write(f"Topics related to '{search_word}':")
        st.write(search_results)

    # # Calculate coherence score
    # dictionary = Dictionary(df['processed_text'])
    # corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
    # coherence_model = CoherenceModel(topics=[topic_model.get_topic(i) for i in range(num_topics)], texts=df['processed_text'], dictionary=dictionary, coherence='c_v')
    # coherence_score = coherence_model.get_coherence()
    # st.write(f"Coherence Score: {coherence_score}")

    # Perplexity score calculation using sklearn's CountVectorizer
    vectorizer = CountVectorizer()
    transformed_documents = vectorizer.fit_transform(documents)



    # # Visualize topics with word clouds
    # for topic_id in range(topic_model.get_number_of_topics()):
    #     st.write(f"Topic {topic_id + 1}")
    #     words = topic_model.get_topic(topic_id)[:10]  # Get top 10 words per topic
    #     create_wordcloud([word for word, _ in words], f"Topic {topic_id + 1}")

    # num_topics = len(set(topics))  # Determine the number of unique topics
    # for topic_id in range(num_topics):
    #     st.write(f"Topic {topic_id + 1}")
    #     words = topic_model.get_topic(topic_id)[:10]  # Get top 10 words per topic
    #     create_wordcloud([word for word, _ in words], f"Topic {topic_id + 1}")


    # Calculate coherence score
    dictionary = Dictionary(df['processed_text'])
    corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
    coherence_model = CoherenceModel(topics=topic_model.get_topics(), texts=df['processed_text'], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    st.write(f"Coherence Score: {coherence_score}")

    # Perplexity score calculation using sklearn's CountVectorizer
    vectorizer = CountVectorizer()
    transformed_documents = vectorizer.fit_transform(documents)
    perplexity_score = topic_model.perplexity(transformed_documents)

    st.write(f"Perplexity Score: {perplexity_score}")

    # Search for topics
    search_word = st.text_input("Enter a word to find related topics:")
    if search_word:
        search_topics, search_probabilities = topic_model.find_topics(search_word)
        search_results = pd.DataFrame({
            'Topic': search_topics,
            'Probability': search_probabilities
        }).sort_values(by='Probability', ascending=False)
        st.write(f"Topics related to '{search_word}':")
        st.write(search_results)
