import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    else:
        return []

def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

st.title("NLP Preprocessing and Topic Modeling App")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform topic modeling to visualize the topics using BERTopic.
""")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        text_column = st.selectbox("Select the text column for preprocessing", df.columns)

        df['processed_text'] = df[text_column].apply(preprocess_text)

        st.write("Preprocessed Text Head:")
        st.write(df[['processed_text']].head())

        df = df.dropna(subset=['processed_text'])

        documents = df['processed_text'].apply(lambda x: ' '.join(x))

        try:
            from bertopic import BERTopic
            import hdbscan

            topic_model = BERTopic()
            topics, probabilities = topic_model.fit_transform(documents)

            st.write("Topic Visualization:")
            st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

            search_word = st.text_input("Enter a word to find related topics:")
            if search_word:
                search_topics, search_probabilities = topic_model.find_topics(search_word)
                search_results = pd.DataFrame({
                    'Topic': search_topics,
                    'Probability': search_probabilities
                }).sort_values(by='Probability', ascending=False)
                st.write(f"Topics related to '{search_word}':")
                st.write(search_results)

            num_topics = len(set(topics))
            for topic_id in range(num_topics):
                st.write(f"Topic {topic_id + 1}")
                words = topic_model.get_topic(topic_id)[:10]
                create_wordcloud([word for word, _ in words], f"Topic {topic_id + 1}")

        except ImportError as e:
            st.error(f"Dependency error: {e}. Please ensure all dependencies are properly installed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
