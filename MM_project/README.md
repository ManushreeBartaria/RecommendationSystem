Fashion Recommendation System
About the Project
This project recommends fashion products to users based on their interests, preferences, and past activity.

The goal is simple: help users find products they are more likely to like.

The project uses product details, user preferences, and interaction history to create personalized recommendations.
How It Works
1. Product information: The system reads details such as product category, brand, color, material, style, price, and rating.
2. User activity: It uses events such as viewing, clicking, wishlisting, adding to cart, purchasing, or marking a product as not interested.
3. Content-based filtering: TF-IDF converts product information into numerical features so the system can identify products with similar descriptions and attributes.
4. Collaborative filtering: Truncated SVD finds patterns in user–product interactions to help recommend products based on interaction patterns.
5. Hybrid recommendations: The system combines content similarity, collaborative-filtering scores, and product popularity to rank products.
6. Streamlit dashboard: The app displays recommendations and lets users explore products and interaction data.
Technologies Used
- Python — main programming language
- Pandas and NumPy — data handling
- Scikit-learn — TF-IDF and Truncated SVD
- SciPy — sparse user–product matrix
- Joblib — saving and loading model artifacts
- Streamlit — interactive web app
Data Files
The project uses three CSV files:
- users.csv — user profiles and preferences
- products.csv — product details
- interaction_events.csv — user activity with products
The trained model components are saved in:
- data/fashion_recommender_artifacts.joblib
The Streamlit app loads these saved artifacts, so it does not need to train the content and collaborative models each time it starts.
Project Structure
fashion-recommender/
├── app.py
└── data/
    ├── users.csv
    ├── products.csv
    ├── interaction_events.csv
    └── fashion_recommender_artifacts.joblib
How to Run the Project
1. Install the required packages
pip install streamlit pandas numpy scikit-learn scipy joblib
2. Check the files
Make sure app.py is in the project folder and the three CSV files and model artifact are in the data/ folder.
3. Start the app
Run this command from the project folder:
streamlit run app.py
Streamlit will provide a local URL that you can open in your browser.
Main Features
- View an overview of the datasets
- Explore popular products
- Find products similar to a selected product
- Get recommendations for a selected user
- Explore user interaction analytics
- Adjust the contribution of content, collaborative filtering, and popularity scores (if enabled in the app)
Recommendation Approach
The app uses a hybrid recommendation approach:
- Content score: How closely a product matches relevant product details or a user's interests
- Collaborative score: How well a product fits patterns learned from user–product interactions
- Popularity score: How popular a product is based on positive interactions
These scores are combined to produce a ranked list of products. The exact ranking depends on the saved model, the available data, and the settings used in the app.
Important Note
The model artifacts are a saved snapshot of the data used during training. New interaction events do not automatically retrain the model. If the product catalog or interaction data changes and you want the model to reflect those changes, you may need to rebuild and export the artifacts from the training notebook.
Short Project Explanation
My project is a fashion recommendation system that suggests products based on product details, user preferences, and past interactions. I use TF-IDF for content-based filtering and Truncated SVD for collaborative filtering. I combine these signals with product popularity to rank recommendations. The trained model components are saved and loaded into a Streamlit dashboard, where users can explore products and personalized recommendations.