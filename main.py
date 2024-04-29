from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


dataset = pd.read_csv(r"D:/recommend purfume/final_perfume_data.csv", encoding= 'unicode_escape')

# Load the model
similarity_matrix = joblib.load("similarity_matrix.pkl")

# Define FastAPI app
app = FastAPI()

# Define request body schema
class InputData(BaseModel):
    band1: str
# liked_perfumes = ['Tihota Eau de Parfum']
# Recommendation function
def recommend_perfumes(liked_perfumes, similarity_matrix, top_n=5):
    # Assuming dataset is available or you can load it similarly to similarity_matrix
    # Find the indices of liked perfumes
    liked_indices = dataset[dataset['Name'].isin(liked_perfumes)].index

    # Calculate the similarity scores for the liked perfumes
    similarity_scores = similarity_matrix[liked_indices]

    # Calculate the average similarity scores, excluding the liked perfumes
    average_similarity = similarity_scores.mean(axis=0)
    average_similarity[liked_indices] = 0  # Set similarity scores of liked perfumes to 0

    # Get the indices of top n similar perfumes
    top_indices = average_similarity.argsort()[::-1][:top_n]

    # Get the names of recommended perfumes
    recommended_perfumes = dataset.loc[top_indices, 'Name'].tolist()
    
    percentage = []
    for i in similarity_matrix[top_indices]:
        str(percentage.append(round(100 - (abs(np.sum(similarity_scores) - np.sum(i))/np.sum(similarity_scores))*100,2)))
    
    
    return recommended_perfumes, percentage


print(recommend_perfumes(['Sola Parfum'],similarity_matrix))

@app.get("/{x}")
async def recommend(x):
    recommended_perfumes = recommend_perfumes([x], similarity_matrix)
    
    # ทำสิ่งที่ต้องการกับ band1 และส่งคำตอบกลับ
    return {"recommended_perfumes": recommended_perfumes}


# print(recommend_perfumes(liked_perfumes,similarity_matrix))
# Recommendation endpoint
# @app.post("/recommend")
# async def recommend(data: InputData):
#     # Call recommendation function to get recommendations
#     recommended_perfumes = recommend_perfumes([data.band1], similarity_matrix)
    
#     # Return recommendations
#     return {"recommended_perfumes": recommended_perfumes}