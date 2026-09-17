import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# Charger la FAQ(le fichier excel)
df = pd.read_excel("fichier_fusionne_nettoye.xlsx")
df = df.drop_duplicates(subset=['question', 'réponse'])

questions = df['question'].tolist()
reponses = df['réponse'].tolist()

# Générer les embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
question_embeddings = model.encode(questions, convert_to_tensor=True)

os.makedirs("models", exist_ok=True)


data = {
    "questions": questions,
    "reponses": reponses,
    "embeddings": question_embeddings
}

torch.save(data, "models/faq_data.pt")
print("Fichier models/faq_data.pt sauvegardé avec succès.")
