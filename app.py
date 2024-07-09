from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('IndianFoodDatasetCSV.csv')

# Preprocess the 'Ingredients' column to handle missing values
df['TranslatedIngredients'].fillna('', inplace=True)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer and transform the ingredients column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['TranslatedIngredients'])

# Define a function to extract ingredients from a string using regular expressions
def extract_ingredients(input_str):
    return re.findall(r'[^,]+', input_str)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    search_type = request.form.get('search_type')
    if search_type == 'by_ingredients':
        ingredients_str = request.form['ingredients']
        ingredients = extract_ingredients(ingredients_str)
        input_tfidf = tfidf_vectorizer.transform(ingredients)
        
        cosine_similarities = linear_kernel(input_tfidf, tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[:-11:-1]
        related_indices = [idx for idx in related_indices if idx < len(df)]
        
        if not related_indices:
            recommended_recipes = []
        else:
            recommended_recipes = [(df.iloc[idx]['TranslatedRecipeName'], df.iloc[idx]['URL'], df.iloc[idx]['TranslatedInstructions'], df.iloc[idx]['Diet']) for idx in related_indices]
    
    elif search_type == 'by_cuisine':      
        cuisine = request.form['cuisine']
        input_tfidf = tfidf_vectorizer.transform([cuisine])
        
        cosine_similarities = linear_kernel(input_tfidf, tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[:-11:-1]
        related_indices = [idx for idx in related_indices if idx < len(df)]
        
        if not related_indices:
            recommended_recipes = []
        else:
            recommended_recipes = [(df.iloc[idx]['TranslatedRecipeName'], df.iloc[idx]['URL'], df.iloc[idx]['TranslatedInstructions'], df.iloc[idx]['Diet']) for idx in related_indices]
    
    elif search_type == 'by_diet':
        selected_diet = request.form['diet']
        # Randomly select 10 recipes based on the selected diet
        diet_recipes = df[df['Diet'] == selected_diet].sample(n=10)
        recommended_recipes = diet_recipes[['TranslatedRecipeName', 'URL', 'TranslatedInstructions', 'Diet']].values.tolist()
        
    else:
        return "Invalid search type"
    
    import random

    placeholder_images = ["/static/image/placeholder.jpg", "/static/image/placeholder2.jpg", "/static/image/placeholder3.jpg",
                          "/static/image/placeholder4.jpg", "/static/image/placeholder5.jpg", "/static/image/placeholder6.jpg",
                          "/static/image/placeholder7.jpg", "/static/image/placeholder8.jpg", "/static/image/placeholder9.jpg",
                          "/static/image/placeholder10.jpg", "/static/image/placeholder11.jpg", "/static/image/placeholder12.jpg",
                          "/static/image/placeholder13.jpg", "/static/image/placeholder14.jpg", "/static/image/placeholder15.jpg",]

    updated_recipes = []
    for recipe in recommended_recipes:
        updated_recipe = list(recipe[:3]) + [random.choice(placeholder_images)] + [recipe[3]]
        updated_recipes.append(updated_recipe)

    return render_template('index.html', ingredients=ingredients_str if search_type == 'by_ingredients' else "", recommended_recipes=updated_recipes)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
