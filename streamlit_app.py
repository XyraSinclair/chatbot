import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import json
import os

# Initialize OpenAI client
openai_api_key = st.secrets.openai_api_key
client = OpenAI(api_key=openai_api_key)

# Define the geometric mean PCM solver
def solve_pcm_geometric_mean(pcm):
    """
    Solves a complete pairwise comparison matrix using the geometric mean method.
    
    Args:
        pcm: A square matrix of pairwise comparison ratios
        
    Returns:
        A normalized vector of weights
    """
    n = len(pcm)
    weights = np.ones(n)
    
    # Calculate the geometric mean of each row
    for i in range(n):
        row_product = 1.0
        for j in range(n):
            row_product *= pcm[i][j]
        
        weights[i] = row_product ** (1.0 / n)
    
    # Normalize the weights to sum to 1
    return weights / np.sum(weights)

# Cache configuration
CACHE_FILE = "llm_ratings_cache.json"


def load_cache():
    """Load the cache from file or return empty cache if file doesn't exist"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save the cache to file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_cache_key(i, j):
    """Generate a consistent cache key for tweet pair comparison"""
    return f"{i}_{j}"


def get_cached_comparison(cache, attribute, i, j):
    """Get a cached comparison if it exists"""
    if attribute not in cache:
        return None
    cache_key = get_cache_key(i, j)
    return cache[attribute].get("tweet_pairs", {}).get(cache_key)


# Load tweets and attributes from JSON file
def load_data():
    """Load the tweets and attributes from the data.json file"""
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            tweets = data.get('tweets', [])
            attributes_data = data.get('attributes', [])
            
            # Convert attributes to the format expected by the app
            attributes = [(attr['name'], attr['prompt']) for attr in attributes_data]
            
            return tweets, attributes
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], []

# Load the data
tweets, attributes = load_data()

# Show title and description
st.title("LLM Pairwise Ratio Comparison Analysis")
st.write("How well can LLMs put meaningful numbers on stuff across interesting subjective attributes?")

# Store results in session state and preload with example data
if "attribute_scores" not in st.session_state:
    # Create example 4x4 ratio comparison matrix for demonstration
    def create_example_matrices():
        # Choose first 4 tweets to keep it manageable
        demo_tweets = tweets[:4]
        n = len(demo_tweets)
        example_matrices = {}
        
        # Create example matrices for the first 3 attributes
        for attr_name, _ in attributes[:3]:
            # Initialize matrix with 1's on diagonal
            matrix = [[1.0 for _ in range(n)] for _ in range(n)]
            
            # Fill upper triangle with sample values
            # These are sample ratios that make mathematical sense
            if attr_name == attributes[0][0]:  # First attribute
                matrix[0][1], matrix[0][2], matrix[0][3] = 2.0, 1.5, 3.0
                matrix[1][2], matrix[1][3] = 0.75, 1.5
                matrix[2][3] = 2.0
            elif attr_name == attributes[1][0]:  # Second attribute
                matrix[0][1], matrix[0][2], matrix[0][3] = 0.5, 1.0, 0.33
                matrix[1][2], matrix[1][3] = 2.0, 0.67
                matrix[2][3] = 0.33
            else:  # Third attribute
                matrix[0][1], matrix[0][2], matrix[0][3] = 1.0, 0.5, 1.5
                matrix[1][2], matrix[1][3] = 0.5, 1.5
                matrix[2][3] = 3.0
            
            # Fill lower triangle with reciprocals
            for i in range(n):
                for j in range(i+1, n):
                    matrix[j][i] = 1.0 / matrix[i][j] if matrix[i][j] != 0 else 0
            
            # Store in example matrices
            example_matrices[attr_name] = matrix
        
        return example_matrices
    
    # Initialize with example data
    st.session_state.attribute_scores = create_example_matrices()

# Get a default attribute to always show
default_attr = next(iter(st.session_state.attribute_scores.keys()))

# Layout for the main display - move matrix and weights to top
st.header("Pairwise Comparison Matrix")

col1, col2 = st.columns([3, 2])

with col1:
    # Get matrix for default attribute (practicality) to display upfront
    default_matrix = st.session_state.attribute_scores[default_attr]
    
    # Create labels for the matrix
    n_default = len(default_matrix)
    default_labels = [f"Tweet {i+1}" for i in range(n_default)]
    
    # Display the matrix
    st.subheader(f"Matrix for '{default_attr}'")
    default_df = pd.DataFrame(default_matrix, index=default_labels, columns=default_labels)
    st.dataframe(default_df.style.format("{:.2f}"))

with col2:
    # Calculate and display the weights for the default attribute
    default_matrix_np = np.array(default_matrix)
    default_weights = solve_pcm_geometric_mean(default_matrix_np)
    
    # Create weight dataframe
    st.subheader("Weights (Geometric Mean)")
    default_weight_df = pd.DataFrame({
        'Tweet': [f"Tweet {i+1}" for i in range(len(default_weights))],
        'Weight': default_weights
    })
    st.dataframe(default_weight_df.sort_values('Weight', ascending=False).style.format({'Weight': '{:.4f}'}))

# Layout for the controls section
control_col1, control_col2 = st.columns([2, 1])

with control_col1:
    # Select attribute to analyze
    selected_attribute = st.selectbox("Select attribute to analyze:",
                                     [attr[0] for attr in attributes],
                                     key="attribute_selector")
    
    # Find the corresponding prompt for the selected attribute
    selected_prompt = next(
        (prompt for name, prompt in attributes if name == selected_attribute), "")
        
with control_col2:
    # Button to run analysis for the selected attribute 
    st.write("") # Add space for alignment
    st.write("")
    if st.button("Analyze Selected Attribute", use_container_width=True):
        with st.spinner(f"Analyzing '{selected_attribute}'..."):
            analyze_selected_attribute()

# Display tweets for reference
with st.expander("View All Tweets"):
    # Add a numerical ID to each tweet for easier display
    numbered_tweets = {f"Tweet {i+1}": tweet for i, tweet in enumerate(tweets)}
    for tweet_id, tweet_text in numbered_tweets.items():
        st.markdown(f"**{tweet_id}:** {tweet_text}")

# Option to upload custom data
with st.expander("Upload Custom Data"):
    st.write("You can upload your own JSON file with tweets and attributes.")
    st.write("The file should follow this format:")
    st.code('''
{
  "tweets": [
    "Tweet text 1",
    "Tweet text 2",
    ...
  ],
  "attributes": [
    {
      "name": "attribute_name_1",
      "prompt": "comparison prompt for this attribute"
    },
    ...
  ]
}
    ''')
    
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            new_tweets = data.get('tweets', [])
            new_attributes_data = data.get('attributes', [])
            
            if new_tweets and new_attributes_data:
                # Convert attributes to the format expected by the app
                new_attributes = [(attr['name'], attr['prompt']) for attr in new_attributes_data]
                
                # Update the global variables
                tweets = new_tweets
                attributes = new_attributes
                
                # Clear existing scores to prevent mismatches
                st.session_state.attribute_scores = {}
                
                st.success(f"Loaded {len(tweets)} tweets and {len(attributes)} attributes successfully!")
            else:
                st.error("Invalid JSON format. Make sure it contains 'tweets' and 'attributes' arrays.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Divider before additional results
st.markdown("---")
st.header("Selected Attribute Analysis")


# Define the analysis function
def analyze_selected_attribute():
    """Calculate the pairwise comparison matrix for just the selected attribute"""
    attribute_name = selected_attribute
    attribute_prompt = selected_prompt

    # Load cache
    cache = load_cache()

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    comparison_matrix = []
    n = len(tweets)

    # Create empty n×n matrix
    for i in range(n):
        comparison_matrix.append([0] * n)
        # Set diagonal to 1 (self-comparison)
        comparison_matrix[i][i] = 1

    # Calculate total number of comparisons needed
    total_comparisons = n * (n - 1) // 2
    comparisons_done = 0

    # Initialize cache structure for this attribute if it doesn't exist
    if attribute_name not in cache:
        cache[attribute_name] = {"tweet_pairs": {}}

    # Fill the upper triangular part of the matrix
    for i in range(n):
        for j in range(i+1, n):
            # Check cache first
            cached_ratio = get_cached_comparison(cache, attribute_name, i, j)

            if cached_ratio is not None:
                ratio = cached_ratio
                status_text.text(
                    f"Using cached comparison for Tweet {i+1} with Tweet {j+1}")
            else:
                tweet_a = tweets[i]
                tweet_b = tweets[j]

                # Update progress
                status_text.text(f"Comparing Tweet {i+1} with Tweet {j+1}...")

                # Get LLM response for comparison
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that evaluates relative properties of texts. "
                         "Respond with a single numerical ratio. For example, if the first text is "
                         "twice as practical as the second, respond with '2'. If it's half as practical, "
                         "respond with '0.5'. If they're equal, respond with '1'."},
                        {"role": "user", "content": f"{attribute_prompt}\n\nTweet A: {tweet_a}\n\nTweet B: {tweet_b}"}
                    ],
                    temperature=0.2,
                    max_tokens=10
                )

                try:
                    ratio = float(response.choices[0].message.content.strip())
                except ValueError:
                    ratio = 1

                # Cache the new comparison
                cache[attribute_name]["tweet_pairs"][get_cache_key(
                    i, j)] = ratio
                cache[attribute_name]["tweet_pairs"][get_cache_key(
                    j, i)] = 1 / ratio if ratio != 0 else 0

                # Save cache after each new comparison
                save_cache(cache)

            # Fill both positions in the matrix
            comparison_matrix[i][j] = ratio
            comparison_matrix[j][i] = 1 / ratio if ratio != 0 else 0

            # Update progress
            comparisons_done += 1
            progress_bar.progress(comparisons_done / total_comparisons)

    # Clear status text when done
    status_text.empty()
    progress_bar.empty()

    # Store result in session state
    st.session_state.attribute_scores[attribute_name] = comparison_matrix

    return comparison_matrix


# Display results for the selected attribute
st.subheader(f"Analysis for '{selected_attribute}'")

# Get the matrix for the selected attribute if available, otherwise use first available attribute
if selected_attribute in st.session_state.attribute_scores:
    matrix = st.session_state.attribute_scores[selected_attribute]
    display_attr_name = selected_attribute
else:
    # If not analyzed yet, show notice
    matrix = st.session_state.attribute_scores[default_attr]
    display_attr_name = default_attr
    st.info(f"Select an attribute and click 'Analyze Selected Attribute' to see results for that attribute.")

# Display results in a two-column layout
col1, col2 = st.columns([3, 2])

# Limit label count to matrix size
n_tweets_to_display = min(len(matrix), len(tweets))
display_tweets = tweets[:n_tweets_to_display]

# Left column: Display the comparison matrix
with col1:
    # Create labels for the matrix
    labels = [f"Tweet {i+1}" for i in range(n_tweets_to_display)]
    
    # Convert to pandas DataFrame for better display
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    
    # Display the matrix
    st.dataframe(df.style.format("{:.2f}"))
    
    st.caption("""
    **How to read this matrix:** Each cell (i,j) shows how much more of the attribute Tweet i has compared to Tweet j. 
    Values > 1 mean Tweet i has more, values < 1 mean less. Diagonal = 1, and cell (j,i) = 1/cell(i,j).
    """)

# Convert to numpy array and solve using geometric mean
matrix_np = np.array(matrix)
weights = solve_pcm_geometric_mean(matrix_np)

# Right column: Display weights
with col2:
    n_weights = len(weights)  # Get the number of weights
    weight_df = pd.DataFrame({
        'Tweet': [f"Tweet {i+1}" for i in range(n_weights)],
        'Weight': weights
    })
    st.dataframe(weight_df.sort_values('Weight', ascending=False).style.format({'Weight': '{:.4f}'}))
    
    st.caption("""
    **About these weights:** Computed using geometric mean method (nth root of row products, normalized). 
    Higher weights indicate better performance on this attribute.
    """)

# Display ranked tweets 
st.subheader(f"Tweets Ranked by {display_attr_name.title()}")

# Create a top 3 summary view
top_indices = np.argsort(-weights)[:3]  # Get indices of top 3
ranked_col1, ranked_col2, ranked_col3 = st.columns(3)

with ranked_col1:
    if len(top_indices) > 0:
        idx = top_indices[0]
        st.markdown(f"**🥇 First Place (Weight: {weights[idx]:.4f})**")
        st.markdown(f"*Tweet {idx+1}:* {display_tweets[idx][:100]}..." if len(display_tweets[idx]) > 100 else f"*Tweet {idx+1}:* {display_tweets[idx]}")

with ranked_col2:
    if len(top_indices) > 1:
        idx = top_indices[1]
        st.markdown(f"**🥈 Second Place (Weight: {weights[idx]:.4f})**")
        st.markdown(f"*Tweet {idx+1}:* {display_tweets[idx][:100]}..." if len(display_tweets[idx]) > 100 else f"*Tweet {idx+1}:* {display_tweets[idx]}")

with ranked_col3:
    if len(top_indices) > 2:
        idx = top_indices[2]
        st.markdown(f"**🥉 Third Place (Weight: {weights[idx]:.4f})**")
        st.markdown(f"*Tweet {idx+1}:* {display_tweets[idx][:100]}..." if len(display_tweets[idx]) > 100 else f"*Tweet {idx+1}:* {display_tweets[idx]}")

# Show all tweets in expanders
with st.expander("See all ranked tweets"):
    sorted_indices = np.argsort(-weights)  # Negative to sort descending
    
    for rank, idx in enumerate(sorted_indices):
        if idx < len(display_tweets):
            st.markdown(f"**Rank {rank+1} (Weight: {weights[idx]:.4f}):** {display_tweets[idx]}")

# Always show summary table with weights (example data is preloaded)
st.header("Attribute Weight Summary (Geometric Mean Method)")
st.write("This table shows the calculated weights for each tweet across all analyzed attributes using the geometric mean method.")

# Create a DataFrame with tweets as rows
summary_data = []
analyzed_attributes = list(st.session_state.attribute_scores.keys())

# Only process the first 4 tweets for the preloaded example
display_tweets = tweets[:4] if len(tweets) > 4 else tweets

# For each tweet, collect its weights across attributes
for i, tweet in enumerate(display_tweets):
    row_data = {'Tweet ID': f'Tweet {i+1}'}
    
    # First 30 characters of the tweet for better identification
    row_data['Tweet Preview'] = tweet[:30] + "..." if len(tweet) > 30 else tweet
    
    # Add weights for each analyzed attribute
    for attr in analyzed_attributes:
        matrix = st.session_state.attribute_scores[attr]
        matrix_np = np.array(matrix)
        n = len(matrix_np)
        
        # Calculate weights using geometric mean method
        weights = solve_pcm_geometric_mean(matrix_np)
        
        # Store the weight for this tweet and attribute
        if i < len(weights):
            row_data[attr] = weights[i]
    
    summary_data.append(row_data)

# Create and display the DataFrame
summary_df = pd.DataFrame(summary_data)

# Style for better display
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: #d4f1d4' if v else '' for v in is_max]

# Apply styling to numeric columns only
style_cols = analyzed_attributes

st.dataframe(
    summary_df.style
    .format({col: '{:.4f}' for col in analyzed_attributes})
    .apply(highlight_max, subset=style_cols)
)

# Add explanation
st.info("""
**How to read this table:**
- Each row represents a tweet
- Each column represents an attribute
- Values show the calculated weights (higher is better)
- Green highlighting indicates the highest-scoring tweet for each attribute
""")

# Add download button for the summary table
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(summary_df)

st.download_button(
    label="Download Summary Table as CSV",
    data=csv,
    file_name="tweet_attribute_weights.csv",
    mime="text/csv",
)

# Footer with additional information
st.markdown("---")
st.caption("This application demonstrates LLM-based pairwise ratio comparisons for text evaluation.")
st.caption("Results are cached to disk for faster loading of previously analyzed attributes.")
