from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

responses = [
    "Hi there! How can I help you?",
    "I’m doing great, thanks for asking!",
    "The weather is nice today.",
    "Goodbye! Have a nice day!",
    "I can answer simple questions."
]

vectorizer = TfidfVectorizer()

def chatbot(user_input):
	all_text = responses + [user_input]
	tfidf = vectorizer.fit_transform(all_text)

	similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

	idx = similarity.argmax()

	return responses[idx]

if __name__ == '__main__':
	print("Chatbot: Hello! Ask me something.")
	for _ in range(3):
		user = input("You: ")
		print("Chatbot: ", chatbot(user))
