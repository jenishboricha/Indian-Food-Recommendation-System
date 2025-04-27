import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class ClusteringEnsemble:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.models = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
            'gmm': GaussianMixture(n_components=n_clusters, random_state=42),
            'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        self.cluster_labels = {}

    def fit(self, X):
        for name, model in self.models.items():
            try:
                self.cluster_labels[name] = model.fit_predict(X)
            except Exception as e:
                print(f"Warning: {name} clustering failed: {str(e)}")
                continue

    def get_consensus_clusters(self):
        valid_labels = [labels for labels in self.cluster_labels.values()
                        if labels is not None]
        if not valid_labels:
            return np.zeros(self.n_clusters)
        n_samples = len(valid_labels[0])
        final_labels = np.zeros(n_samples)

        for i in range(n_samples):
            sample_labels = [labels[i] for labels in valid_labels]
            final_labels[i] = max(set(sample_labels), key=sample_labels.count)

        return final_labels

class IndianFoodRecommender:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
            self.prepare_features()
            self.train_models()
        except Exception as e:
            print(f"Error initializing recommender: {str(e)}")
            raise

    def prepare_features(self):
        taste_mapping = {'high': 3, 'medium': 2, 'low': 1}

        for col in self.df.columns:
            if col in ['spice_level', 'sweet_level', 'salty_level', 'sour_level', 'bitter_level']:
                self.df[col] = self.df[col].replace(taste_mapping).fillna(0)
            elif self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(0)
            else:
                self.df[col] = self.df[col].fillna('')

        self.df['popularity'] = pd.to_numeric(self.df['popularity'], errors='coerce').fillna(0)

        self.taste_features = ['spice_level', 'sweet_level', 'salty_level',
                               'sour_level', 'bitter_level']

        if 'calories_per_100g' in self.df.columns:
            self.taste_features.append('calories_per_100g')

        self.category_features = [col for col in self.df.columns
                                  if col.startswith(('diet_type_', 'course_'))]

        self.scaler = StandardScaler()
        self.scaled_taste = self.scaler.fit_transform(self.df[self.taste_features])

        self.tfidf = TfidfVectorizer(stop_words='english')
        ingredients_text = self.df['ingredients'].astype(str)
        self.ingredients_matrix = self.tfidf.fit_transform(ingredients_text)

        try:
            self.combined_features = np.hstack([
                self.scaled_taste,
                self.df[self.category_features].fillna(0).values,
                self.ingredients_matrix.toarray()
            ])
        except Exception as e:
            print(f"Error combining features: {str(e)}")
            self.combined_features = self.scaled_taste  # fallback

    def train_models(self):
        try:
            self.cluster_ensemble = ClusteringEnsemble(n_clusters=5)
            self.cluster_ensemble.fit(self.scaled_taste)
            self.consensus_clusters = self.cluster_ensemble.get_consensus_clusters()

            self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.knn.fit(self.ingredients_matrix)

            if 'course_maincourse' in self.df.columns:
                self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
                self.rf.fit(self.scaled_taste, self.df['course_maincourse'].fillna(0))

            if 'popularity' in self.df.columns:
                self.gbm = GradientBoostingRegressor(random_state=42)
                self.gbm.fit(self.scaled_taste, self.df['popularity'].fillna(0.5))
        except Exception as e:
            print(f"Error training models: {str(e)}")

    def get_recommendations(self, liked_dishes, target_state, n_recommendations=2):
        if not isinstance(liked_dishes, list):
            liked_dishes = [liked_dishes]

        recommendations = []
        for dish in liked_dishes:
            try:
                if dish.lower() not in self.df['name'].str.lower().values:
                    raise IndexError(f"Dish '{dish}' not found in database.")

                dish_idx = self.df[self.df['name'].str.lower() == dish.lower()].index[0]

                state_mask = self.df['state'].str.lower() == target_state.lower()
                state_dishes = self.df[state_mask]

                if len(state_dishes) == 0:
                    print(f"No dishes found for state: {target_state}. Using top popular dishes instead.")
                    state_dishes = self.df.nlargest(n_recommendations, 'popularity')

                state_indices = state_dishes.index

                taste_sim = cosine_similarity(
                    self.scaled_taste[dish_idx].reshape(1, -1),
                    self.scaled_taste[state_indices]
                )

                ingredient_sim = cosine_similarity(
                    self.ingredients_matrix[dish_idx],
                    self.ingredients_matrix[state_indices]
                )

                cluster_sim = np.zeros(len(state_indices))
                if self.consensus_clusters is not None:
                    dish_cluster = self.consensus_clusters[dish_idx]
                    state_clusters = self.consensus_clusters[state_indices]
                    cluster_sim[state_clusters == dish_cluster] = 1

                combined_sim = (
                    0.4 * taste_sim[0] +
                    0.4 * ingredient_sim[0] +
                    0.2 * cluster_sim
                )

                sorted_indices = combined_sim.argsort()[::-1]
                top_indices = [idx for idx in sorted_indices if state_dishes.iloc[idx]['name'].lower() != dish.lower()][:n_recommendations]

                if len(top_indices) < n_recommendations:
                    fallback_dishes = self.df.nlargest(n_recommendations, 'popularity')
                    for _, rec_dish in fallback_dishes.iterrows():
                        recommendations.append({
                            'original_dish': dish,
                            'recommended_dish': rec_dish['name'],
                            'similarity_score': 0,
                            'reasoning': "Fallback to popular dishes."
                        })
                else:
                    for idx in top_indices:
                        rec_dish = state_dishes.iloc[idx]
                        recommendations.append({
                            'original_dish': dish,
                            'recommended_dish': rec_dish['name'],
                            'similarity_score': combined_sim[idx],
                            'reasoning': self._generate_reasoning(dish, rec_dish)
                        })

            except IndexError:
                print(f"Warning: Dish '{dish}' not found in database. Using top popular dishes as fallback.")
                fallback_dishes = self.df.nlargest(n_recommendations, 'popularity')
                for _, rec_dish in fallback_dishes.iterrows():
                    recommendations.append({
                        'original_dish': dish,
                        'recommended_dish': rec_dish['name'],
                        'similarity_score': 0,
                        'reasoning': "Fallback to popular dishes."
                    })
            except Exception as e:
                print(f"Error processing dish '{dish}': {str(e)}")
                continue

        return recommendations

    def _generate_reasoning(self, original_dish, recommended_dish):
        try:
            orig = self.df[self.df['name'].str.lower() == original_dish.lower()].iloc[0]
            rec = recommended_dish

            reasons = []

            for taste in self.taste_features:
                if abs(orig[taste] - rec[taste]) < 1:
                    reasons.append(f"similar {taste.replace('_', ' ')}")

            try:
                orig_ingredients = set(orig['ingredients'].split(', '))
                rec_ingredients = set(rec['ingredients'].split(', '))
                common_ingredients = orig_ingredients.intersection(rec_ingredients)

                if len(common_ingredients) > 0:
                    reasons.append(f"common ingredients: {', '.join(common_ingredients)}")
            except:
                pass

            return " | ".join(reasons)
        except Exception as e:
            print(f"Error generating reasoning: {str(e)}")
            return "Similar taste or course."


st.title("Indian Food Recommender System 🍛")
st.write("Get personalized food recommendations based on your favorite Indian dishes!")

csv_file_path = "/home/jenish/ML/ml_mini_project/Indian Food Recommendation/indian_food_dataset.csv" 

try:
    recommender = IndianFoodRecommender(csv_file_path)

    states = sorted(recommender.df['state'].unique())

    st.subheader("Enter Your Preferences")

    available_dishes = sorted(recommender.df['name'].unique())
    liked_dishes = st.multiselect(
        "Select your favorite dishes:",
        options=available_dishes,
        help="You can select multiple dishes"
    )

    selected_state = st.selectbox(
        "Select a state for recommendations:",
        options=states,
        help="Choose the state you want recommendations from"
    )

    n_recommendations = st.slider(
        "Number of recommendations per dish:",
        min_value=1,
        max_value=5,
        value=2,
        help="How many recommendations would you like for each dish?"
    )

    if st.button("Get Recommendations"):
        if liked_dishes:
            with st.spinner("Generating recommendations..."):
                recommendations = recommender.get_recommendations(
                    liked_dishes,
                    selected_state,
                    n_recommendations
                )

            st.subheader("Your Personalized Recommendations")

            for rec in recommendations:
                with st.container():
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(f"**Original Dish:**")
                        st.write(rec['original_dish'])

                    with col2:
                        st.markdown(f"**Recommended Dish:**")
                        st.write(rec['recommended_dish'])

                    similarity_percentage = float(rec['similarity_score']) * 100
                    st.markdown("**Similarity Score:**")
                    st.progress(rec['similarity_score'])
                    st.write(f"{similarity_percentage:.1f}%")

                    st.markdown("**Why this recommendation?**")
                    st.write(rec['reasoning'])

                    st.divider()
        else:
            st.warning("Please select at least one favorite dish.")

except Exception as e:
    st.error(f"Error initializing the recommender system: {str(e)}")

