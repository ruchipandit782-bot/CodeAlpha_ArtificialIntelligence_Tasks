import json
import re
import math
import numpy as np

class FAQChatbot:
    def __init__(self, faqs_path):
        self.faqs = []
        self.vocab = []
        self.idf = {}
        self.faq_vectors = None
        self.load_faqs(faqs_path)

    def preprocess(self, text):
        """
        Tokenizes text, converts to lowercase, and removes punctuation.
        """
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words

    def load_faqs(self, path):
        """
        Loads FAQs from JSON and fits the TF-IDF representation on questions.
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.faqs = json.load(f)
            
        # Extract questions and tokenize
        corpus = [self.preprocess(faq['question']) for faq in self.faqs]
        
        # Build vocabulary
        all_words = []
        for doc in corpus:
            all_words.extend(doc)
        self.vocab = sorted(list(set(all_words)))
        
        # Calculate IDF
        N = len(corpus)
        for word in self.vocab:
            df = sum(1 for doc in corpus if word in doc)
            # IDF with smoothing
            self.idf[word] = math.log((1 + N) / (1 + df)) + 1
            
        # Vectorize FAQ questions
        self.faq_vectors = np.array([self.vectorize(doc) for doc in corpus])

    def vectorize(self, doc):
        """
        Converts a preprocessed document into a TF-IDF vector.
        """
        vector = np.zeros(len(self.vocab))
        if not doc:
            return vector
            
        # Term Frequency
        tf = {}
        for word in doc:
            tf[word] = tf.get(word, 0) + 1
            
        # Calculate TF-IDF for words in vocabulary
        for idx, word in enumerate(self.vocab):
            if word in tf:
                # TF normalized by doc length * IDF
                vector[idx] = (tf[word] / len(doc)) * self.idf[word]
        return vector

    def get_response(self, user_query, threshold=0.15):
        """
        Matches user query against FAQs and returns the best matching answer.
        """
        query_tokens = self.preprocess(user_query)
        if not query_tokens:
            return "Please type a valid question."

        query_vector = self.vectorize(query_tokens)
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return "I'm sorry, I don't understand that question. Could you try rephrasing it?"

        # Compute cosine similarity for all FAQ vectors
        similarities = []
        for faq_vector in self.faq_vectors:
            faq_norm = np.linalg.norm(faq_vector)
            if faq_norm == 0:
                similarities.append(0.0)
                continue
            
            # dot product / (norm1 * norm2)
            sim = np.dot(faq_vector, query_vector) / (faq_norm * query_norm)
            similarities.append(sim)

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        print(f"Query: '{user_query}' | Best match: '{self.faqs[best_idx]['question']}' | Score: {best_score:.4f}")

        if best_score >= threshold:
            return self.faqs[best_idx]['answer']
        else:
            return "I'm sorry, I couldn't find a close match for that question in my database. Can you try asking differently?"
