# main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import threading
import time
import httpx
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Load MongoDB URI from .env (never hardcode credentials)
URI = os.getenv("MONGO_URI")
if not URI:
    raise ValueError("MONGO_URI not set in environment variables")

client = MongoClient(URI)
db = client["Placementdb"]
collection = db["users"]

app = FastAPI(
    title="VCET Placement Top Students API",
    description="Returns top N students based on academics/LeetCode or placement similarity",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local dev
        "https://your-frontend.onrender.com",  # Replace with your actual deployed frontend URL
        # Add more if needed, e.g. Vercel/Netlify URL
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/top-students")
def get_top_students(
    n: int = Query(20, ge=1, le=100, description="Number of top students to return (1-100)")
):
    # Fetch all students
    students = list(collection.find())

    if not students:
        return {"error": "No students found in database"}

    # Build DataFrame
    data = []
    for student in students:
        data.append({
            "username": student.get("username", "Unknown"),
            "x_percent": student.get("X % ", 0),
            "xii_percent": student.get("XII % ", 0),
            "ug_aggregate": student.get("Aggregate UG", 0),
            "easy": student.get("EASY", 0),
            "medium": student.get("MEDUIM", 0),
            "hard": student.get("HARD", 0),
            "placements": len(student.get("companies", []))
        })

    df = pd.DataFrame(data)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Normalize academics (0-1)
    scaler_minmax = MinMaxScaler()
    df[["x_percent", "xii_percent", "ug_aggregate"]] = scaler_minmax.fit_transform(
        df[["x_percent", "xii_percent", "ug_aggregate"]]
    )

    # Weighted LeetCode score
    df["leetcode_score"] = df["easy"] * 1 + df["medium"] * 2 + df["hard"] * 3

    # Features for KNN
    features = ["x_percent", "xii_percent", "ug_aggregate", "leetcode_score", "placements"]
    df_features = df[features].copy()
    scaler_std = StandardScaler()
    df_features_scaled = pd.DataFrame(scaler_std.fit_transform(df_features), columns=features)

    # Phase 1: Weighted academic + LeetCode score
    df["weighted_score"] = (
        df["x_percent"] * 0.15 +
        df["xii_percent"] * 0.15 +
        df["ug_aggregate"] * 0.30 +
        df["leetcode_score"] * 0.40
    )

    # Check placement status
    placed_students = df[df["placements"] >= 1]

    # Use Phase 2 only if at least 10% of students have placements
    if len(placed_students) < (0.1 * len(df)):
        top_df = df.sort_values("weighted_score", ascending=False).head(n)
        return {
            "phase": 1,
            "message": "Not enough placements yet — using academic + LeetCode ranking",
            "count": len(top_df),
            "top_students": top_df[["username", "weighted_score"]].round(4).to_dict(orient="records")
        }
    else:
        # Phase 2: KNN similarity to placed students
        placed_features = df_features_scaled.loc[placed_students.index]
        knn = NearestNeighbors(n_neighbors=min(5, len(placed_features)), metric="euclidean")
        knn.fit(placed_features)

        distances, _ = knn.kneighbors(df_features_scaled)
        df["knn_distance"] = distances.mean(axis=1)
        df["norm_similarity"] = 1 / (1 + df["knn_distance"])
        df["norm_placements"] = MinMaxScaler().fit_transform(df[["placements"]])

        df["final_score"] = (0.6 * df["norm_placements"]) + (0.4 * df["norm_similarity"])

        top_df = df.sort_values("final_score", ascending=False).head(n)
        return {
            "phase": 2,
            "message": "Using placement similarity (KNN) ranking",
            "count": len(top_df),
            "top_students": top_df[["username", "final_score", "placements"]].round(4).to_dict(orient="records")
        }

@app.get("/")
def home():
    return {"message": "VCET Placement Ranking Microservice is running!"}

  # We can use httpx (already installed with FastAPI) instead of requests

# Optional self-ping to reduce cold starts on Render
def start_keep_alive():
    # Get the external URL from environment (set in Render dashboard)
    external_url = os.getenv("RENDER_EXTERNAL_URL")
    if not external_url:
        print("No RENDER_EXTERNAL_URL set — skipping self-ping (cold starts may occur)")
        return

    ping_url = f"{external_url.rstrip('/')}/top-students?n=1"  # Fast, lightweight call

    def ping_loop():
        while True:
            try:
                httpx.get(ping_url, timeout=10)
                print(f"Self-ping sent to {ping_url}")
            except Exception as e:
                print(f"Self-ping failed: {e}")
            time.sleep(30)  # Every 5 minutes (300 seconds)

    # Run in background thread
    thread = threading.Thread(target=ping_loop, daemon=True)
    thread.start()
    print("Keep-alive self-ping started (every 5 minutes)")

# Start the keep-alive when the app loads
start_keep_alive()