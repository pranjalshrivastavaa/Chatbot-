import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Initialize NLP Tools
lemmatizer = WordNetLemmatizer()

# Load dataset from JSON file
try:
    with open("data.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    data = {
    "greetings": ["hello", "hi", "hey", "howdy", "hola", "good morning", "good evening", "good afternoon"],
    "responses": ["Hello!", "Hi there!", "Hey!", "Greetings!", "Nice to see you!", "Good to have you here!"],
    
    "farewell": ["bye", "goodbye", "see you", "take care", "catch you later"],
    "farewell_responses": ["Goodbye!", "See you soon!", "Take care!", "Have a great day!", "Bye!"],

    "small_talk": ["how are you?", "what's up?", "how's it going?", "how have you been?"],
    "small_talk_responses": ["I'm just a bot, but I'm doing great!", "I'm good, thanks for asking!", "Just here to chat with you!"],

    "jokes": ["tell me a joke", "make me laugh", "say something funny"],
    "jokes_responses": ["Why don’t scientists trust atoms? Because they make up everything!", "Parallel lines have so much in common. It’s a shame they’ll never meet."],

    "motivation": ["motivate me", "tell me something inspiring", "I need motivation"],
    "motivation_responses": ["You are capable of amazing things!", "Believe in yourself, and you will succeed!", "Your future is created by what you do today."],

    "weather": ["what's the weather like?", "is it raining?", "is it sunny?", "how's the weather today?"],
    "weather_responses": ["I'm not connected to the internet, but I hope it's nice!", "Check a weather app for real-time updates!"]
}

def preprocess(sentence):
    tokens = sentence.lower().split()  # Use split() instead of NLTK tokenizer
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return " ".join(tokens)


# Generate a response based on user input
def get_response(user_input):
    user_input = preprocess(user_input)

    all_phrases = []
    intent_map = {}

    # Create a mapping of all phrases to intents
    for intent, phrases in data.items():
        for phrase in phrases:
            processed_phrase = preprocess(phrase)
            all_phrases.append(processed_phrase)
            intent_map[processed_phrase] = intent

    if not all_phrases:
        return "I'm sorry, I don't have enough data to respond."

    # Convert text to vectors for similarity matching
    from sklearn.feature_extraction.text import TfidfVectorizer
 
    vectorizer = TfidfVectorizer().fit_transform([user_input] + all_phrases)

    vectors = vectorizer.toarray()

    # Compute cosine similarity
    similarity_scores = cosine_similarity([vectors[0]], vectors[1:])[0]

    # Get the most similar phrase
    best_match_index = similarity_scores.argmax()
    confidence = similarity_scores[best_match_index]

    # Set a confidence threshold for accurate responses
    if confidence > 0.4:  # Adjust threshold as needed
        best_match_phrase = all_phrases[best_match_index]
        intent_category = intent_map[best_match_phrase]
        return random.choice(data[intent_category + "_responses"]) if intent_category + "_responses" in data else random.choice(data[intent_category])

    return "I'm sorry, I don't understand."

# Chat function
def chat():
    print("Chatbot: Hello! I'm your friendly chatbot. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        response = get_response(user_input)
        print("Chatbot:", response)

# Run chatbot
if __name__ == "__main__":
    chat()
