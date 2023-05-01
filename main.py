import streamlit as st
import os
from googletrans import Translator 
import spacy
import nltk
from nltk import sent_tokenize
from textblob import TextBlob
from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

nlph = spacy.load('xx_ent_wiki_sm')
nlpe = spacy.load('en_core_web_sm')
#---------------------------functions-------------------------

#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList
# Function for Sumy Summarization

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result
# Function to Analyse Tokens and Lemma

def text_analyzer(my_text):    
    docx = nlpe(my_text)
    # tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData

def entity_analyzer(my_text):
    entities = []
    entityLabels = []
    doc = nlph(my_text)
    for ent in doc.ents:
        entities.append(ent.text)
        entityLabels.append(ent.label_)
    entDict = dict(zip(entities, entityLabels))

    entOrg = entRecognizer(entDict, "ORG")
    entCardinal = entRecognizer(entDict, "CARDINAL")
    entPerson = entRecognizer(entDict, "PERSON")
    entDate = entRecognizer(entDict, "DATE")
    entGPE = entRecognizer(entDict, "GPE")

    entity_types = {
        "Organization Entities": entOrg,
        "Cardinal Entities": entCardinal,
        "Person Entities": entPerson,
        "Date Entities": entDate,
        "GPE Entities": entGPE
    }
    return entity_types
# Function For Extracting title From Text
def extract_title(text):
    # Using Spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    
    # Using TextBlob
    sentences = sent_tokenize(text)
    for sentence in sentences:
        blob = TextBlob(sentence)
        if blob[0].isupper() and blob[1:].islower():
            return sentence.strip()
    # Using NLTK
    tagged_sentences = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(tagged_sentences)
    for subtree in tree.subtrees():
        if subtree.label() == 'KT':
            return ' '.join(word for word, tag in subtree.leaves())
    
    return None
# function for tanslation 

def translate_text(text, lang):
    translator = Translator()
    if lang == 'Hindi':
        translated_text = translator.translate(text, src='en', dest='hi')
        return translated_text.text
    elif lang == 'Marathi':
        translated_text = translator.translate(text, src='en', dest='mr')
        return translated_text.text
    else:
        return "Invalid language selection"
    # ------------------------------------------------------hindi text function ----------------------------

def hindi_text_analyzer(my_text):
    docx = nlph(my_text)
    tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData
def hindi_entity_analyzer(my_text):
    doc = nlph(my_text)
    entities = []
    entity_labels = []
    for ent in doc.ents:
        entities.append(ent.text)
        entity_labels.append(ent.label_)
    return entities, entity_labels


def hindisentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

# ----------------------- main function--------------------------------

def main():
    # title 
    st.title("Text Analysis Tool")
    st.subheader("Natural Language Processing On the Go")
    st.sidebar.subheader("About the App")
    st.sidebar.text("NLP for everyone.")
    st.sidebar.info("Use this tool to get the sentiment score, tokens , lemma, Named Entities and Summary of your text. It's the ultimate!")
    #langauge options
    language = st.sidebar.selectbox("Select Language", ("English", "Hindi"))
    #text input
    if language == "English":
            st.subheader("English Text")
            raw_text = st.text_area("Your Text")
            # tokenization
            if st.checkbox("Show Tokens and Lemma"):
                st.subheader("Tokenize Your Text")
                # if st.button("Analyze"):
                nlp_result = text_analyzer(raw_text)
                st.write(nlp_result)
            # Named Entity
            if st.checkbox("Show Named Entities"):
                    st.subheader("Analyze Your Text")
                # if st.button("Extract"):
                    entity_result = entity_analyzer(raw_text)
                    st.write(entity_result)
            # Sentiment Analysis
            if st.checkbox("Show Sentiment Analysis"):
                st.subheader("Sentiment of Your Text")
                # if st.button("Analyze"):
                blob = TextBlob(raw_text)
                result_sentiment = blob.sentiment
                st.write(result_sentiment)
            # Summarization
            if st.checkbox("Show Text Summarization"):
                st.subheader("Summarize Your Text")
                summary_options = st.selectbox("Choose Summarizer", ("gensim", "sumy"))
                if summary_options == "gensim":
                    st.text("Using Gensim Summarizer ..")
                    # if st.button("Summarize"):
                    st.text("Using Gensim Summarizer ..")
                    summary_result = summarize(raw_text)
                    st.write(summary_result)
                elif summary_options == "sumy":
                    st.text("Using Sumy Summarizer ..")
                    # if st.button("Summarize"):
                    st.text("Using Sumy Summarizer ..")
                    summary_result = sumy_summarizer(raw_text)
                    st.write(summary_result)
            # Translation
            if st.checkbox("Show Translation"):
                st.subheader("Translate Your Text")
                lang = st.selectbox("Choose Language", ("Hindi", "Marathi"))
                # if st.button("Translate"):
                translated_text = translate_text(raw_text, lang)
                st.write(translated_text)
            # Title
            if st.checkbox("Show Title"):
                st.subheader("Title of Your Text")
                # if st.button("Extract"):
                title_result = extract_title(raw_text)
                st.write(title_result)
    elif language == "Hindi":
            st.subheader("Hindi Text")
            raw_text = st.text_area("Your Text")

            # tokenization
            if st.checkbox("Show Tokens and Lemma"):
                st.subheader("Tokenize Your Text")
                # if st.button("Analyze"):
                nlp_result = hindi_text_analyzer(raw_text)
                st.write(nlp_result)
            # Named Entity
            if st.checkbox("Show Named Entities"):
                st.subheader("Analyze Your Text")
                # if st.button("Extract"):
                entity_result = hindi_entity_analyzer(raw_text)
                st.write(entity_result)
            # Sentiment Analysis
            if st.checkbox("Show Sentiment Analysis"):
                st.subheader("Sentiment of Your Text")
                # if st.button("Analyze"):
                sentiment_result = hindisentiment_analysis(raw_text)
                st.write(sentiment_result)
            # # Summarization
            # if st.checkbox("Show Text Summarization"):
            #     st.subheader("Summarize Your Text")
            #     summary_options = st.selectbox("Choose Summarizer", ("gensim", "sumy"))
            #     if summary_options == "gensim":
            #         st.text("Using Gensim Summarizer ..")
            #         if st.button("Summarize"):
            #             st.text("Using Gensim Summarizer ..")
            #             summary_result = summarize(raw_text)
            #             st.write(summary_result)
            #     elif summary_options == "sumy":
            #         st.text("Using Sumy Summarizer ..")
            #         if st.button("Summarize"):
            #             st.text("Using Sumy Summarizer ..")
            #             summary_result = sumy_summarizer(raw_text)
            #             st.write(summary_result)
            # # Translation
            # if st.checkbox("Show Translation"):
            #     st.subheader("Translate Your Text")
            #     lang = st.selectbox("Choose Language", ("English", "Marathi"))
            #     if st.button("Translate"):
            #         translated_text = translate_text(raw_text, lang)
            #         st.write(translated_text)
            # # Title
            # if st.checkbox("Show Title"):
            #     st.subheader("Title of Your Text")
            #     if st.button("Extract"):
            #         title_result = extract_title(raw_text)
            #         st.write(title_result)
    # # Keywords

if __name__ =='__main__':
    main()




