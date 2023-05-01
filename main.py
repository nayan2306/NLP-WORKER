
# Core Pkgs
import streamlit as st
import os
from googletrans import Translator
# NLP Pkgs
from nltk import sent_tokenize
from textblob import TextBlob
import spacy
from gensim.summarization.summarizer import summarize
import nltk
nltk.download('punkt')
# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList
# Function for Sumy Summarization
@st.cache
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result
# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData
@st.cache
def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    entities = []
    entityLabels = []
    doc = nlp(my_text)
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
        "Personal Entities": entPerson,
        "Date Entities": entDate,
        "GPE Entities": entGPE,
    }

    return entity_types
@st.cache
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
@st.cache
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

def main():
	""" NLP Based App with Streamlit """
	# Title
	st.title("Ultimate NLP Application")
	st.subheader("Natural Language Processing for everyone")
	st.markdown("""
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
    	Tokenization , Lemmatization, Named Entity Recognition (NER), Sentiment Analysis, Text Summarization. Click any of the checkboxes to get started.
    	""")
	# Summarization
	if st.checkbox("Get the summary of your text"):
		st.subheader("Summarize Your Text")

		message = st.text_area("Enter Text","Type Here....")
		summary_options = st.selectbox("Choose Summarizer",['sumy','gensim'])
		if st.button("Summarize"):
			if summary_options == 'sumy':
				st.text("Using Sumy Summarizer ..")
				summary_result = sumy_summarizer(message)
			elif summary_options == 'gensim':
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)
			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim Summarizer ..")
				summary_result = summarize(message)
			st.success(summary_result)
	# Sentiment Analysis
	if st.checkbox("Get the Sentiment Score of your text"):
		st.subheader("Identify Sentiment in your Text")

		message = st.text_area("Enter Text","Type Here...")
		if st.button("Analyze"):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(result_sentiment)
	# Entity Extraction
	if st.checkbox("Get the Named Entities of your text"):
		st.subheader("Identify Entities in your text")

		message = st.text_area("Enter Text","Type Here..")
		if st.button("Extract"):
			entity_result = entity_analyzer(message)
			st.write(entity_result)
	# Tokenization
	if st.checkbox("Get the Tokens and Lemma of text"):
		st.subheader("Tokenize Your Text")

		message = st.text_area("Enter Text","Type Here.")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.write(nlp_result)
	# extract_title
	if st.checkbox("Get the title of your text"):
		st.subheader("Extract Title of your text")
		message = st.text_area("Enter Text","Type Here.")
		if st.button("Extract"):
			title_result = extract_title(message)
			st.write(title_result)
	# Translation
	if st.checkbox("Translate your text"):
		st.subheader("Translate your text")
		message = st.text_area("Enter Text","Type Here.")
		lang = st.selectbox("Choose Language",['Hindi','Marathi'])
		if st.button("Translate"):
			translation_result = translate_text(message, lang)
			st.write(translation_result)
	


	st.sidebar.subheader("About the App")
	st.sidebar.text("NLP for everyone.")
	st.sidebar.info("Use this tool to get the sentiment score, tokens , lemma, Named Entities and Summary of your text. It's the ultimate!")



if __name__ == '__main__':
	main()
