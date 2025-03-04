import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import json
import os
import re

# Initialize OpenAI client
openai_api_key = st.secrets.openai_api_key
client = OpenAI(api_key=openai_api_key)

# Configure the page
st.set_page_config(
    page_title="Consistency Checker",
    page_icon="🧮",
    layout="wide"
)

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

# Calculate inconsistency ratio
def calculate_consistency(pcm):
    """
    Calculate the consistency ratio of a pairwise comparison matrix
    
    Args:
        pcm: A square numpy matrix of pairwise comparison ratios
        
    Returns:
        consistency_ratio: A measure of the inconsistency of the matrix
        perfect_matrix: An ideally consistent matrix based on the eigenvector weights
    """
    n = len(pcm)
    
    # Calculate weights using geometric mean method
    weights = solve_pcm_geometric_mean(pcm)
    
    # Create a perfectly consistent matrix based on the weights
    perfect_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            perfect_matrix[i, j] = weights[i] / weights[j]
    
    # Calculate the log differences for each cell
    log_diffs = []
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle
            if pcm[i, j] > 0 and perfect_matrix[i, j] > 0:
                log_diff = abs(np.log10(pcm[i, j]) - np.log10(perfect_matrix[i, j]))
                log_diffs.append(log_diff)
    
    # Calculate the mean log difference
    if log_diffs:
        consistency_score = np.mean(log_diffs)
    else:
        consistency_score = 0
    
    return consistency_score, perfect_matrix

# Cache configuration
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_file():
    """Get the cache file path"""
    return os.path.join(CACHE_DIR, "consistency_cache.json")

def load_cache():
    """Load the cache from file or return empty cache if file doesn't exist"""
    cache_file = get_cache_file()
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save the cache to file"""
    cache_file = get_cache_file()
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def get_cache_key(prompt_hash, items_hash):
    """Generate a cache key for a set of items and comparison prompt"""
    return f"{prompt_hash}_{items_hash}"

# Load default tweets from data.json
def load_default_data():
    """Load the tweets from the data.json file"""
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            tweets = data.get('tweets', [])
            return tweets[:5]  # Just take 5 items for simplicity
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [
            "meditation and mindfulness practices",
            "reading philosophy books",
            "going for long walks in nature",
            "engaging in deep conversations",
            "journaling about personal insights"
        ]

# Initialize session state
if "items" not in st.session_state:
    st.session_state["items"] = load_default_data()

if "attribute_prompt" not in st.session_state:
    st.session_state["attribute_prompt"] = "How much more utilitous is entity A than entity B for the typical Bay Area tpot rationalist?"

if "num_samples" not in st.session_state:
    st.session_state["num_samples"] = 3

if "matrix" not in st.session_state:
    st.session_state["matrix"] = None

if "consistency_score" not in st.session_state:
    st.session_state["consistency_score"] = None

if "perfect_matrix" not in st.session_state:
    st.session_state["perfect_matrix"] = None

if "variance_matrix" not in st.session_state:
    st.session_state["variance_matrix"] = None

def query_llm_for_item_comparisons(items, attribute_prompt, temperature=0.3, num_samples=None):
    """
    Query the LLM to do pairwise comparisons of items for a specific attribute
    Returns a pairwise comparison matrix for the items and a variance matrix
    
    Args:
        items: List of items to compare
        attribute_prompt: The prompt to use for comparison
        temperature: Temperature setting for the LLM (default: 0.3)
        num_samples: Number of times to query per pair (uses session state if None)
    """
    # Use the provided num_samples or get from session state
    if num_samples is None:
        num_samples = st.session_state["num_samples"]
    
    # Generate a unique hash for this set of items, prompt, and sample count
    items_str = json.dumps(items)
    items_hash = hash(items_str)
    prompt_hash = hash(attribute_prompt)
    samples_hash = hash(str(num_samples))
    
    # Load cache
    cache = load_cache()
    cache_key = get_cache_key(f"{prompt_hash}_{samples_hash}", items_hash)
    
    if cache_key in cache:
        st.success("Using cached comparisons")
        # Return both matrices from cache
        return np.array(cache[cache_key]["pcm"]), np.array(cache[cache_key]["variance"])
    
    st.info(f"Comparing items with {num_samples} samples per pair...")
    
    n = len(items)
    pcm = np.ones((n, n))
    variance_matrix = np.zeros((n, n))
    
    # Calculate total comparisons (including multiple samples)
    total_comparisons = n * (n - 1) // 2 * num_samples
    progress_bar = st.progress(0)
    status_text = st.empty()
    comparisons_done = 0
    
    try:
        # For each pair of items, ask LLM to compare
        for i in range(n):
            for j in range(i+1, n):
                item_a = items[i]
                item_b = items[j]
                
                # Use letters (A, B, C...) instead of numbers
                label_i = chr(65 + i)
                label_j = chr(65 + j)
                
                # Collect multiple samples for this pair
                ratios = []
                
                for sample_idx in range(num_samples):
                    status_text.text(f"Comparing Entity {label_i} with Entity {label_j} (Sample {sample_idx+1}/{num_samples})...")
                    
                    prompt = f"""{attribute_prompt}

Entity A: {item_a}

Entity B: {item_b}

Respond with a single numerical ratio. For example, if A is 2.5 as good as B, respond with "2". 
If A is half as good as B, respond with "0.5". If they're equal, respond with "1"."""

                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that evaluates relative properties of items. You MUST respond with ONLY a numerical ratio, with no explanations or additional text. Just return the number."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=50
                    )
                    
                    try:
                        # First attempt to parse directly
                        raw_response = response.choices[0].message.content.strip()
                        try:
                            ratio = float(raw_response)
                        except ValueError:
                            # If direct parsing fails, make a second LLM call to extract just the number
                            extraction_prompt = f"""Extract ONLY the numerical ratio from this response. Return ONLY the number, with no other text:

{raw_response}

For example, if the response contains "I think A is 2.5 times better than B", just return "2.5".
If the response indicates equality, return "1".
If B is better than A, return a decimal like "0.5".
"""

                            extraction_response = client.chat.completions.create(
                                model="gpt-4", 
                                messages=[
                                    {"role": "system", "content": "You extract numerical values from text."},
                                    {"role": "user", "content": extraction_prompt}
                                ],
                                temperature=min(0.1, temperature),  # Always keep this very low for extracting numbers
                                max_tokens=10
                            )
                            
                            ratio = float(extraction_response.choices[0].message.content.strip())
                            
                            # Log the extra call that was needed
                            st.info(f"Needed to extract number from: '{raw_response}'")
                            
                    except ValueError as e:
                        st.warning(f"Could not parse response for comparison of entities {label_i} and {label_j}, using default value of 1. Error: {e}")
                        ratio = 1.0
                    
                    # Store this sample's ratio
                    ratios.append(ratio)
                    
                    # Update progress
                    comparisons_done += 1
                    progress_bar.progress(comparisons_done / total_comparisons)
                
                # Calculate the mean and variance
                mean_ratio = np.mean(ratios)
                var_ratio = np.var(ratios) if len(ratios) > 1 else 0.0
                
                # Store the mean and variance
                pcm[i, j] = mean_ratio
                pcm[j, i] = 1.0 / mean_ratio if mean_ratio != 0 else 0
                
                variance_matrix[i, j] = var_ratio
                variance_matrix[j, i] = var_ratio  # Store same variance for both directions
                
                # Progress is already updated inside the sample loop
        
        # Clear status text and progress bar when done
        status_text.empty()
        progress_bar.empty()
        
        # Cache the results with both matrices
        cache[cache_key] = {
            "pcm": pcm.tolist(),
            "variance": variance_matrix.tolist()
        }
        save_cache(cache)
        
        return pcm, variance_matrix
        
    except Exception as e:
        st.error(f"Error comparing items: {e}")
        # Return identity matrices as fallback
        return np.ones((n, n)), np.zeros((n, n))

def run_analysis():
    """
    Run the consistency analysis for the attribute and items
    """
    items = st.session_state["items"]
    attribute_prompt = st.session_state["attribute_prompt"]
    num_samples = st.session_state["num_samples"]
    
    if not items or len(items) < 2:
        st.error("Please provide at least 2 items to compare")
        return
    
    # Get pairwise comparison matrix and variance matrix
    pcm, variance_matrix = query_llm_for_item_comparisons(
        items, attribute_prompt, num_samples=num_samples
    )
    
    # Calculate consistency
    consistency_score, perfect_matrix = calculate_consistency(pcm)
    
    # Update session state
    st.session_state["matrix"] = pcm
    st.session_state["perfect_matrix"] = perfect_matrix
    st.session_state["consistency_score"] = consistency_score
    st.session_state["variance_matrix"] = variance_matrix
    
    st.success(f"Analysis completed with {num_samples} samples per pair!")

# Main app UI with improved styling
st.title("🧠 Consistency Checker")
st.markdown("""
<div style="background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
This tool tests how <b>logically consistent</b> LLMs are when making pairwise comparisons between different entities.
</div>
""", unsafe_allow_html=True)

# Create a cleaner layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    # Input the attribute prompt
    st.subheader("Comparison Prompt")
    attribute_prompt = st.text_area("", 
                                  value=st.session_state["attribute_prompt"],
                                  help="This prompt will be sent to the LLM for each pairwise comparison",
                                  height=70)
    if attribute_prompt != st.session_state["attribute_prompt"]:
        st.session_state["attribute_prompt"] = attribute_prompt

with col2:
    # Model settings
    st.subheader("Model Settings")
    
    # Number of samples per pair
    num_samples = st.number_input(
        "Samples per pair:",
        min_value=1,
        max_value=10,
        value=st.session_state["num_samples"],
        help="Number of times to query the LLM for each pair to measure variance"
    )
    if num_samples != st.session_state["num_samples"]:
        st.session_state["num_samples"] = num_samples

# Add a divider
st.markdown("---")

# Input entities to evaluate
st.subheader("Entities to Compare")

# Create form for items
with st.form(key="items_form"):
    # Get default items if needed
    default_items = load_default_data()
    
    # Access items safely
    if "items" not in st.session_state or not isinstance(st.session_state["items"], list):
        st.session_state["items"] = default_items
    
    items = st.session_state["items"]
        
    # Create a grid layout for entities
    num_items = len(items)
    rows = (num_items + 1) // 2  # Calculate number of rows, with 2 entities per row
    
    # Create text inputs for each item in a grid
    item_inputs = []
    for row in range(rows):
        cols = st.columns(2)  # 2 columns per row
        for col in range(2):
            i = row * 2 + col
            if i < num_items:
                # Use A, B, C... as labels
                label = chr(65 + i)  # ASCII: A=65, B=66, etc.
                with cols[col]:
                    item_inputs.append(
                        st.text_area(f"Entity {label}", value=items[i], height=70, key=f"item_{i}")
                    )
    
    # Add/remove entity buttons with better styling
    st.markdown("#### Actions")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        add_item = st.form_submit_button("➕ Add Entity", use_container_width=True)
    with col2:
        remove_item = st.form_submit_button("➖ Remove Last", use_container_width=True)
    with col3:
        # Form submission button with prominent styling
        analyze_button = st.form_submit_button("▶️ Run Consistency Analysis", 
                                            use_container_width=True,
                                            type="primary")

# Handle form actions
if add_item:
    st.session_state["items"].append("New Entity")
    st.rerun()
    
if remove_item and len(st.session_state["items"]) > 1:
    st.session_state["items"].pop()
    st.rerun()

# Update items from input fields
for i, key in enumerate([f"item_{j}" for j in range(len(st.session_state["items"]))]):
    if key in st.session_state:
        st.session_state["items"][i] = st.session_state[key]

# Run analysis if button was clicked
if analyze_button:
    run_analysis()

# Show results if analysis has been run
if "matrix" in st.session_state and st.session_state["matrix"] is not None:
    st.header("Analysis Results")
    
    # Get the data
    matrix = st.session_state["matrix"]
    perfect_matrix = st.session_state["perfect_matrix"]
    consistency_score = st.session_state["consistency_score"]
    items = st.session_state["items"]
    
    # Create nice labels using letters (A, B, C...)
    labels = [chr(65 + i) for i in range(len(items))]
    
    # Show consistency score with explanation
    score_color = "green" if consistency_score < 0.1 else "orange" if consistency_score < 0.2 else "red"
    st.markdown(f"### Consistency Score: <span style='color:{score_color}'>{consistency_score:.3f}</span>", unsafe_allow_html=True)
    
    # Interpret the score
    if consistency_score < 0.1:
        st.success("The LLM's judgments are highly consistent!")
    elif consistency_score < 0.2:
        st.warning("The LLM's judgments have some inconsistencies.")
    else:
        st.error("The LLM's judgments are significantly inconsistent.")
        
    # Explain the consistency score calculation
    st.info("""
    **What this score means:**
    
    The consistency score measures how well the LLM's pairwise comparisons align with perfect mathematical consistency.
    
    - **0.0** would be perfect consistency (impossible in practice)
    - **< 0.1** indicates excellent consistency
    - **0.1 - 0.2** indicates acceptable consistency
    - **> 0.2** indicates concerning inconsistency
    
    *Technical explanation:* This score is calculated as the average logarithmic difference between the actual pairwise ratios and what they would be in a perfectly consistent matrix derived from the calculated weights. Smaller values indicate better consistency.
    """)
    
    # Calculate weights using geometric mean method
    weights = solve_pcm_geometric_mean(matrix)
    
    # Create unified results display
    st.subheader("Comparison Matrix with Consistency Analysis")
    
    # Display settings used for this analysis
    st.markdown(f"**Analysis Settings:** Samples per pair = {num_samples}")
    
    # If we have variance information, display it
    if "variance_matrix" in st.session_state and st.session_state["variance_matrix"] is not None:
        with st.expander("View Variance Analysis"):
            # Calculate average variance
            variance_matrix = st.session_state["variance_matrix"]
            avg_variance = np.mean(variance_matrix[np.triu_indices_from(variance_matrix, k=1)])
            
            st.markdown(f"**Average Variance:** {avg_variance:.4f}")
            st.info("This represents how much the LLM's judgments varied across multiple samples for each pair. Lower values indicate more consistent responses.")
            
            # Create a visualization of the variance matrix
            variance_df = pd.DataFrame(variance_matrix, index=labels, columns=labels)
            
            # Define a function to highlight high variance
            def highlight_variance(val):
                if i == j:  # Skip diagonal
                    return 'background-color: #f0f0f0'
                    
                if val > 0.5:
                    return 'background-color: #ff9999'  # Red for high variance
                elif val > 0.2:
                    return 'background-color: #ffcc99'  # Orange for medium variance
                elif val > 0.05:
                    return 'background-color: #ffffcc'  # Yellow for low variance
                elif val > 0:
                    return 'background-color: #ccffcc'  # Green for very low variance
                else:
                    return 'background-color: #f0f0f0'  # Gray for zero variance
            
            # Display the variance matrix
            st.subheader("Variance Matrix")
            st.dataframe(
                variance_df.style.format("{:.4f}").applymap(lambda x: f"background-color: {'#ff9999' if x > 0.5 else '#ffcc99' if x > 0.2 else '#ffffcc' if x > 0.05 else '#ccffcc' if x > 0 else '#f0f0f0'}"),
                use_container_width=True
            )
            
            # Add variance legend
            st.markdown("""
            **Variance Legend:**
            - <span style='background-color: #ccffcc; padding: 2px 5px;'>Green</span>: Very low variance (< 0.05)
            - <span style='background-color: #ffffcc; padding: 2px 5px;'>Yellow</span>: Low variance (0.05 - 0.2)
            - <span style='background-color: #ffcc99; padding: 2px 5px;'>Orange</span>: Medium variance (0.2 - 0.5)
            - <span style='background-color: #ff9999; padding: 2px 5px;'>Red</span>: High variance (> 0.5)
            """, unsafe_allow_html=True)
    
    # Calculate log differences for color coding
    diff_matrix = np.zeros_like(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if perfect_matrix[i, j] > 0 and matrix[i, j] > 0:
                diff_matrix[i, j] = abs(np.log10(matrix[i, j]) - np.log10(perfect_matrix[i, j]))
            else:
                diff_matrix[i, j] = 0
    
    # Create a DataFrame with matrix values
    df_combined = pd.DataFrame(matrix, index=labels, columns=labels)
    
    # Create a DataFrame for weights
    weight_df = pd.DataFrame({
        "Entity": labels,
        "Weight": weights
    }).set_index("Entity")
    
    # Join the matrix with weights column
    df_combined = pd.concat([df_combined, weight_df], axis=1)
    
    # Define the styling function for the combined table
    def color_consistency(df):
        # Make a copy to avoid modifying the original DataFrame
        styled = pd.DataFrame('', index=df.index, columns=df.columns)
        
        # Format and color background based on consistency
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                diff = diff_matrix[i, j]
                
                # Format the number with 3 decimal places
                cell_value = f"{val:.3f}"
                
                # Choose background color based on deviation from ideal
                if i != j:  # Skip diagonal
                    if diff > 0.5:
                        bg_color = '#ff9999'  # Red for large inconsistency
                    elif diff > 0.2:
                        bg_color = '#ffcc99'  # Orange for medium inconsistency
                    elif diff > 0.1:
                        bg_color = '#ffffcc'  # Yellow for small inconsistency
                    else:
                        bg_color = '#ccffcc'  # Green for consistent
                else:
                    bg_color = '#f0f0f0'  # Gray for diagonal
                
                styled.iloc[i, j] = f'background-color: {bg_color}'
                
        # Format the weights column
        for i in range(len(labels)):
            styled.iloc[i, -1] = 'background-color: #e6f3ff; font-weight: bold'
            
        return styled
    
    # Apply styling and display
    st.dataframe(
        df_combined.style
        .format({col: "{:.3f}" for col in labels})
        .format({"Weight": "{:.3f}"})
        .apply(color_consistency, axis=None),
        use_container_width=True
    )
    
    # Add legend for color coding
    st.markdown("""
    **Color Legend:**
    - <span style='background-color: #ccffcc; padding: 2px 5px;'>Green</span>: Highly consistent (< 0.1 difference)
    - <span style='background-color: #ffffcc; padding: 2px 5px;'>Yellow</span>: Slightly inconsistent (0.1 - 0.2 difference)
    - <span style='background-color: #ffcc99; padding: 2px 5px;'>Orange</span>: Moderately inconsistent (0.2 - 0.5 difference)
    - <span style='background-color: #ff9999; padding: 2px 5px;'>Red</span>: Highly inconsistent (> 0.5 difference)
    """, unsafe_allow_html=True)
    
    # Display the weights as a bar chart
    st.subheader("Entity Weights Visualization")
    st.bar_chart(weight_df)
    
    # Optional: Show the tabs for detailed view
    with st.expander("Show Detailed Matrices"):
        tab1, tab2 = st.tabs(["Ideal Consistent Matrix", "Difference Matrix"])
        
        with tab1:
            # Convert to pandas DataFrame for better display
            df_perfect = pd.DataFrame(perfect_matrix, index=labels, columns=labels)
            st.dataframe(df_perfect.style.format("{:.3f}"), use_container_width=True)
            st.info("This matrix shows what a perfectly consistent matrix would look like based on the calculated weights.")
            
        with tab2:
            # Calculate difference matrix
            diff_matrix_display = matrix - perfect_matrix
            
            # Convert to pandas DataFrame for better display
            df_diff = pd.DataFrame(diff_matrix_display, index=labels, columns=labels)
            
            # Custom formatting for differences
            def highlight_diff(val):
                color = 'white'
                if abs(val) > 1.0:
                    color = '#ff9999'  # red
                elif abs(val) > 0.5:
                    color = '#ffcc99'  # orange
                elif abs(val) > 0.1:
                    color = '#ffffcc'  # yellow
                return f'background-color: {color}'
            
            st.dataframe(df_diff.style.format("{:.3f}").applymap(highlight_diff), use_container_width=True)
            st.info("This shows the difference between the LLM's judgments and what would be perfectly consistent judgments.")
    
    # Removed Inconsistency Details section
    
    # Explain inconsistency
    with st.expander("Understanding Consistency in Pairwise Comparisons"):
        st.markdown("""
        ### What is Consistency?
        
        In pairwise comparisons, consistency means that your judgments follow logical transitivity. For example:
        
        - If A is 2 times better than B, and
        - B is 3 times better than C, then
        - A should be 6 times better than C (2 × 3)
        
        ### How to Read the Consistency Score
        
        The consistency score measures how much the LLM's judgments deviate from perfect logical consistency:
        
        - **< 0.1:** Highly consistent judgments
        - **0.1 - 0.2:** Moderately consistent with some inconsistencies
        - **> 0.2:** Significantly inconsistent judgments
        
        ### Why Inconsistency Happens
        
        Inconsistency in LLM judgments can occur because:
        
        1. The LLM may focus on different aspects when making different comparisons
        2. The LLM lacks a global view of all its judgments
        3. Even humans make inconsistent judgments for complex comparisons
        """)

# Footer
st.markdown("---")
st.caption("Consistency Checker - Tests if LLMs make logically consistent pairwise comparison judgments")