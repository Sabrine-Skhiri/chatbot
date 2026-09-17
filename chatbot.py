import pandas as pd
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect

# Charger la FAQ en français 
df = pd.read_excel("fichier_fusionne_nettoye.xlsx")
questions = df['question'].tolist()
reponses = df['réponse'].tolist()

# Embeddings multilingues 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

print("\n🤖 Chatbot FAQ multilingue (1 fichier) — Tapez 'exit' pour quitter\n")

while True:
    user_input = input("👤 Vous : ")
    if user_input.lower() == "exit":
        print("👋 À bientôt !")
        break

    # Détection de la langue
    user_lang = detect(user_input)

    # Embedding/recherche
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score > 0.5:
        reponse_fr = reponses[best_idx]
        # Traduction 
        if user_lang != 'fr':
            try:
                reponse_traduite = GoogleTranslator(source='fr', target=user_lang).translate(reponse_fr)
                print(f"🤖 {reponse_traduite} (traduit du français)\n")
            except Exception as e:
                print(f"🤖 (FR) {reponse_fr} (⚠ erreur de traduction)\n")
        else:
            print(f"🤖 {reponse_fr}\n")
    else:
        print("🤖 Je ne trouve pas de réponse adaptée.\n")