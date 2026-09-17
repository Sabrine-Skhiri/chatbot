import os
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect

# Initialisation Flask 
app = Flask(__name__)

#  Chargement du modèle sauvegardé 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Charger les embeddings avec chemin absolu 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "models", "faq_data.pt")

data = torch.load(DATA_PATH)
questions = data["questions"]
reponses = data["reponses"]
question_embeddings = data["embeddings"]


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Message vide"}), 400

    user_lang = detect(user_input)
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score > 0.5:
        reponse_fr = reponses[best_idx]
        if user_lang != 'fr':
            try:
                reponse_traduite = GoogleTranslator(source='fr', target=user_lang).translate(reponse_fr)
                return jsonify({"response": reponse_traduite, "lang": user_lang, "translated": True})
            except Exception as e:
                return jsonify({"response": reponse_fr, "lang": "fr", "translated": False, "error": str(e)})
        else:
            return jsonify({"response": reponse_fr, "lang": "fr", "translated": False})
    else:
        return jsonify({"response": "Je ne trouve pas de réponse adaptée.", "lang": user_lang})
if __name__ == '__main__':
    app.run(port=5005, debug=True, threaded=True)
