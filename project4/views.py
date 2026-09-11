from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
from collections import Counter
import os
import random
from django.utils import timezone
from .models import StudySession, Interaction
from .preference_model import fit_preference_weights, posterior_covariance

# __file__ based path to the movie metadata CSV file, independent of the current working directory
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'movie_metadata.csv')
# Only load the dataset once when the module is imported, to avoid reloading it on every request
DATASET = pd.read_csv(DATA_PATH)


# Define the numeric features to be used in the feature matrix, excluding certain features that are not relevant for modeling
EXCLUDE_NUMERIC_FEATURES = ['aspect_ratio', 'facenumber_in_poster']
NUMERIC_FEATURES = [col for col in DATASET.select_dtypes(include=['float64', 'int64']).columns.tolist() if col not in EXCLUDE_NUMERIC_FEATURES]

# Define the features that will undergo log transformation to reduce skewness due to Blockbuster movies with very high values, which can disproportionately affect the model
LOG_TRANSFORM_FEATURES = ['gross', 'budget', 'num_voted_users', 'num_user_for_reviews', 'num_critic_for_reviews', 'movie_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes']

# Define constants for the number of top categories to keep for categorical features, with the rest being bundled into an 'Other' category to reduce dimensionality and noise in the feature matrix
TOP_COUNTRIES = 5
TOP_CONTENT_RATINGS = 5
TOP_KEYWORDS = 30


def load_movie_data():
    df = DATASET.copy()
    # Dataset contains invisible characters and extra spaces in the 'movie_title' and 'color' columns, which can cause issues during analysis. These are cleaned up by stripping whitespace and filling missing values with 'Unknown'.
    df['movie_title'] = df['movie_title'].str.strip()
    df['color'] = df['color'].fillna('Unknown').str.strip()
    return df

# Function to bundle rare categories into an 'Other' category
def bundle_rare_categories(series, top_n):
    top_values = series.value_counts().head(top_n).index
    # Replace values not in the top_n with 'Other', while preserving NaN values
    return series.where(series.isin(top_values) | series.isnull(), 'Other')


def get_keyword_list(raw_value):
    if pd.isna(raw_value) or raw_value == '':
        return []
    return raw_value.split('|')


def build_feature_matrix(df):

    feature_frames = []
    color_dummies = pd.get_dummies(df['color'], prefix='color')
    feature_frames.append(color_dummies)


    numeric_part = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        # Create a new column to indicate whether the original value was missing (1 for missing, 0 for present)
        numeric_part[f'{col}_missing'] = df[col].isna().astype(int)
        # Median instead of mean is used to fill missing values to reduce the impact of outliers
        filled = df[col].fillna(df[col].median())

        if col in LOG_TRANSFORM_FEATURES:
            filled = np.log1p(filled)
        
        std = filled.std()
        if std == 0:
            # Avoid division by zero in case of constant columns
            std = 1
        # Standardize the filled numeric values to have a mean of 0 and a standard deviation of 1
        numeric_part[col] = (filled - filled.mean()) / std
    feature_frames.append(numeric_part)

    both_known = df['gross'].notna() & df['budget'].notna()
    gross_budget_ratio = df['gross'] / df['budget']
    median_ratio = gross_budget_ratio[both_known].median()
    gross_budget_ratio = gross_budget_ratio.where(both_known, median_ratio)
    upper_limit = gross_budget_ratio[both_known].quantile(0.99)
    gross_budget_ratio = gross_budget_ratio.clip(upper=upper_limit)
    ratio_mean = gross_budget_ratio.mean()
    ratio_std = gross_budget_ratio.std()
    if ratio_std == 0:
        ratio_std = 1
    gross_budget_ratio = (gross_budget_ratio - ratio_mean) / ratio_std
    feature_frames.append(pd.DataFrame({'gross_budget_ratio': gross_budget_ratio}, index=df.index))

    # Some movies have multiple genres, separated by '|'. Multi-Hot encoding is used to create binary features for each genre
    genre_dummies = df['genres'].str.get_dummies(sep='|').add_prefix('genre_')
    feature_frames.append(genre_dummies)

    # TOP_CONTENT_RATINGS is used to limit the number of content rating categories, bundling less common ratings into an 'Other' category
    rating_bucketed = bundle_rare_categories(df['content_rating'], TOP_CONTENT_RATINGS).fillna('Unknown')
    feature_frames.append(pd.get_dummies(rating_bucketed, prefix='rating'))

    
    country_bucketed = bundle_rare_categories(df['country'], TOP_COUNTRIES).fillna('Unknown')
    feature_frames.append(pd.get_dummies(country_bucketed, prefix='country'))

    keyword_lists = df['plot_keywords'].apply(get_keyword_list)

    all_keywords = []
    for keywords in keyword_lists:
        all_keywords.extend(keywords)
    # Count the occurrences of each keyword across all movies to identify the most common ones
    keyword_counts = Counter(all_keywords)
    top_keywords = []
    for keyword, count in keyword_counts.most_common(TOP_KEYWORDS):
        top_keywords.append(keyword)

    keywords_data = {}
    for keyword in top_keywords:
        column_values = []
        for keywords in keyword_lists:
            if keyword in keywords:
                column_values.append(1)
            else:
                column_values.append(0)
        # Prefix the keyword column names with 'keyword_' to avoid potential naming conflicts and to clearly indicate that these columns represent keyword presence
        keywords_data[f'keyword_{keyword}'] = column_values

    feature_frames.append(pd.DataFrame(keywords_data, index=df.index))

    return pd.concat(feature_frames, axis=1)


# Loaded once at server start. The study needs the same feature matrix on every request
MOVIE_DATA = load_movie_data()
FEATURE_MATRIX = build_feature_matrix(MOVIE_DATA).to_numpy().astype(float)

# Both designs are set to roughly the same number of movies shown per participant
NUMBER_OF_PAIRWISE_TRIALS = 20
NUMBER_OF_RANKING_TRIALS = 5
RANKING_SIZE = 10
STUDY_SESSION_KEY = 'study_session_id'


def number_of_trials(design):
    if design == 'pairwise':
        return NUMBER_OF_PAIRWISE_TRIALS
    return NUMBER_OF_RANKING_TRIALS


def movies_per_trial_for(design):
    if design == 'pairwise':
        return 2
    return RANKING_SIZE


def get_movie_description(movie_index):

    row = MOVIE_DATA.iloc[movie_index]

    year = row['title_year']
    if pd.isna(year):
        year_text = 'year unknown'
    else:
        year_text = str(int(year))

    score = row['imdb_score']
    if pd.isna(score):
        score_text = 'unknown'
    else:
        score_text = str(score)

    director = row['director_name']
    if pd.isna(director):
        director_text = 'director unknown'
    else:
        director_text = director

    genres = row['genres'].replace('|', ', ')

    return {
        'index': int(movie_index),
        'title': row['movie_title'],
        'year': year_text,
        'score': score_text,
        'director': director_text,
        'genres': genres,
    }

# Creates the next question with randomly drawn movies for the given session
def create_next_interaction(study_session):
    already_answered = study_session.interactions.count()
    movies_per_trial = movies_per_trial_for(study_session.design)
    movie_indices_shown = random.sample(range(len(FEATURE_MATRIX)), movies_per_trial)

    return Interaction.objects.create(
        study_session=study_session,
        position=already_answered + 1,
        shown_movie_indices=movie_indices_shown,
        answer_movie_indices=[],
    )


def get_current_study_session(request):
    session_id = request.session.get(STUDY_SESSION_KEY)
    if session_id is None:
        return None
    try:
        return StudySession.objects.get(pk=session_id)
    except StudySession.DoesNotExist:
        return None


def index(request):
    return render(request, 'project4/index.html')


def start_study(request):
    if request.method != 'POST':
        return redirect('project4:index')

    if request.POST.get('consent') != 'yes':
        return render(request, 'project4/index.html', {'consent_missing': True})

    # Random assignment to make designs comparable so that a participant is not able to self-select into the design they prefer, which could bias the results.
    # This ensures that the study can fairly compare the two designs without participants being able to pick the interface they find more convenient
    design = random.choice(['pairwise', 'ranking'])
    study_session = StudySession.objects.create(design=design, consent_given=True)

    request.session[STUDY_SESSION_KEY] = study_session.pk
    create_next_interaction(study_session)
    return redirect('project4:trial')


def trial(request):
    study_session = get_current_study_session(request)
    if study_session is None:
        return redirect('project4:index')

    current_interaction = study_session.interactions.filter(answered_at__isnull=True).first()
    if current_interaction is None:
        return redirect('project4:result')

    if request.method == 'POST':
        chosen_index = int(request.POST.get('chosen_index'))
        chosen_so_far = list(current_interaction.answer_movie_indices)

        if chosen_index in current_interaction.shown_movie_indices and chosen_index not in chosen_so_far:
            chosen_so_far.append(chosen_index)

        remaining = []
        for movie_index in current_interaction.shown_movie_indices:
            if movie_index not in chosen_so_far:
                remaining.append(movie_index)

        # With one movie left its rank is already determined, so the participant
        # is never asked for a click that carries no information.
        if len(remaining) == 1:
            chosen_so_far.append(remaining[0])
            remaining = []

        current_interaction.answer_movie_indices = chosen_so_far
        if len(remaining) == 0:
            current_interaction.answered_at = timezone.now()
        current_interaction.save()

        if len(remaining) > 0:
            return redirect('project4:trial')

        if study_session.interactions.count() < number_of_trials(study_session.design):
            create_next_interaction(study_session)
            return redirect('project4:trial')

        study_session.completed_at = timezone.now()
        study_session.save()
        return redirect('project4:result')

    chosen_so_far = list(current_interaction.answer_movie_indices)

    chosen_movies = []
    for movie_index in chosen_so_far:
        chosen_movies.append(get_movie_description(movie_index))

    remaining_movies = []
    for movie_index in current_interaction.shown_movie_indices:
        if movie_index not in chosen_so_far:
            remaining_movies.append(get_movie_description(movie_index))

    context = {
        'design': study_session.design,
        'trial_number': current_interaction.position,
        'total_trials': number_of_trials(study_session.design),
        'chosen_movies': chosen_movies,
        'remaining_movies': remaining_movies,
    }
    return render(request, 'project4/trial.html', context)


def result(request):
    study_session = get_current_study_session(request)
    if study_session is None:
        return redirect('project4:index')

    rankings = []
    for interaction in study_session.interactions.filter(answered_at__isnull=False):
        rankings.append(list(interaction.answer_movie_indices))

    if len(rankings) == 0:
        return redirect('project4:index')

    weights = fit_preference_weights(rankings, FEATURE_MATRIX)
    covariance = posterior_covariance(weights, rankings, FEATURE_MATRIX)
    # The prior variance is 1.0, so this reads as how much of the initial ignorance about this participant has been resolved
    mean_variance = float(np.mean(np.diag(covariance)))
    certainty_percent = round((1.0 - mean_variance) * 100.0, 1)

    already_seen = []
    for interaction in study_session.interactions.all():
        already_seen.extend(interaction.shown_movie_indices)

    # Recommend the highest-scoring movies the participant has not seen yet
    utilities = FEATURE_MATRIX.dot(weights)
    ranked_positions = np.argsort(-utilities)

    recommendations = []
    for movie_index in ranked_positions:
        if int(movie_index) in already_seen:
            continue
        recommendations.append(get_movie_description(int(movie_index)))
        if len(recommendations) == 10:
            break

    context = {
        'design': study_session.design,
        'number_of_trials': len(rankings),
        'certainty_percent': certainty_percent,
        'recommendations': recommendations,
    }
    return render(request, 'project4/result.html', context)

