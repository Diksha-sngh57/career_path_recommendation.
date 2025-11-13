🎯 AI-Based Career Recommendation System

This project is an AI-powered Career Recommendation System that predicts the best-suited career path for users based on their skills, interests, and education using a Random Forest Classifier and K-Means clustering for pattern discovery.

📁 Project Structure
career_recommendation.py
AI-based Career Recommendation System.csv
career_recommendation_model.pkl
le_skills.pkl
le_interests.pkl
le_education.pkl
le_career.pkl
career_distribution.png
correlation_matrix.png
confusion_matrix.png
clustering_plot.png

🧠 Features

ML-based prediction of suitable careers

Encodes categorical variables (skills, interests, education)

Uses Random Forest Classifier for career prediction

Applies K-Means clustering for visual analysis of user profiles

Automatically generates analytical visualizations

Saves the trained model and encoders for later use

🧩 Technologies Used
Library	Purpose
Pandas, NumPy	Data handling and transformation
Scikit-learn	Machine learning model training and evaluation
Seaborn, Matplotlib	Data visualization
Joblib	Model persistence and encoder saving
KMeans	Clustering and pattern identification
⚙️ How to Run
1️⃣ Clone or Download the Repository
git clone https://github.com/yourusername/ai-career-recommendation.git
cd ai-career-recommendation

2️⃣ Install Dependencies
pip install pandas numpy scikit-learn seaborn matplotlib joblib

3️⃣ Add Your Dataset

Make sure the file AI-based Career Recommendation System.csv is present in the same folder as the script.
It should contain at least the following columns:

Skills

Interests

Education

Recommended_Career

4️⃣ Run the Script
python career_recommendation.py


This will:

Train and evaluate your model

Save trained models and encoders (.pkl files)

Generate multiple plots for visual analysis

🧮 Example: Making Predictions

Once trained, you can predict careers like this:

import joblib
from career_recommendation import recommend_career

# Load model and encoders
model = joblib.load("career_recommendation_model.pkl")
le_skills = joblib.load("le_skills.pkl")
le_interests = joblib.load("le_interests.pkl")
le_education = joblib.load("le_education.pkl")
le_career = joblib.load("le_career.pkl")

# Example prediction
career = recommend_career(
    skills="Python,Data Analysis",
    interests="AI;Machine Learning",
    education="B.Tech",
    experience=2,
    le_skills=le_skills,
    le_interests=le_interests,
    le_education=le_education,
    le_career=le_career,
    rf_model=model
)
print("Recommended Career:", career)

📊 Visual Outputs
File	Description
career_distribution.png	Frequency of different recommended careers
correlation_matrix.png	Correlation between encoded features
confusion_matrix.png	Model performance matrix
clustering_plot.png	K-Means clustering visualization
💾 Saved Files
File	Description
career_recommendation_model.pkl	Trained Random Forest model
le_skills.pkl	Skills encoder
le_interests.pkl	Interests encoder
le_education.pkl	Education encoder
le_career.pkl	Career label encoder
👩‍💻 Author

Diksha Singh
B.Tech in Computer Science & Engineering
Project Guide: Mr. Arpit Mishra

📜 License

This project is developed solely by Diksha Singh for academic and research purposes.
You may modify or extend it for educational or personal learning use.
