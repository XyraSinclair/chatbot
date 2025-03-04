import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import json
import os
import re
import time
from typing import Dict, List, Tuple, Any, Optional

# Initialize OpenAI client
openai_api_key = st.secrets.openai_api_key
client = OpenAI(api_key=openai_api_key)

# Configure the page
st.set_page_config(
    page_title="Analytic Hierarchy Process (AHP) Agent",
    page_icon="📊",
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

# Cache configuration
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_file(prefix=""):
    """Get the cache file path with optional prefix"""
    return os.path.join(CACHE_DIR, f"{prefix}_ahp_cache.json")

def load_cache(prefix=""):
    """Load the cache from file or return empty cache if file doesn't exist"""
    cache_file = get_cache_file(prefix)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, prefix=""):
    """Save the cache to file"""
    cache_file = get_cache_file(prefix)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def get_cache_key(i, j):
    """Generate a consistent cache key for item pair comparison"""
    return f"{i}_{j}"

def get_cached_comparison(cache, attribute, i, j):
    """Get a cached comparison if it exists"""
    if attribute not in cache:
        return None
    cache_key = get_cache_key(i, j)
    return cache[attribute].get("item_pairs", {}).get(cache_key)

# Load default tweets from data.json
def load_default_data():
    """Load the tweets from the data.json file"""
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            tweets = data.get('tweets', [])
            return tweets
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []

# Initialize state for items
if "items" not in st.session_state:
    st.session_state.items = load_default_data()

if "goal" not in st.session_state:
    st.session_state.goal = "Find the relative utilities of me reading these tweets"

if "attributes" not in st.session_state:
    st.session_state.attributes = []

if "attribute_weights" not in st.session_state:
    st.session_state.attribute_weights = {}
    
if "attribute_matrices" not in st.session_state:
    st.session_state.attribute_matrices = {}
    
if "ahp_scores" not in st.session_state:
    st.session_state.ahp_scores = {}
    
if "direct_scores" not in st.session_state:
    st.session_state.direct_scores = {}

def query_llm_for_attributes(items, goal, num_attributes=5):
    """
    Query the LLM to generate attributes for evaluation based on the goal
    """
    cache = load_cache("attributes")
    cache_key = f"{goal}_{len(items)}"
    
    if cache_key in cache:
        st.success("Using cached attributes")
        return cache[cache_key]
    
    # Get a sample of items (if many) to avoid token limits
    sample_items = items[:5] if len(items) > 5 else items
    
    st.info("Generating attributes based on goal...")
    
    try:
        prompt = f"""You are an analytic hierarchy process agent. You are interested in this goal: {goal}.

Below are some sample items that users will be evaluating:

{json.dumps(sample_items, indent=2)}

Based on the goal and these items, come up with {num_attributes} attributes that would be relevant for evaluation. 
For each attribute:
1. Provide a clear name
2. Provide a description
3. Provide a comparison prompt (e.g., "How much more [attribute] is item A compared to item B?")

Format your response as a JSON array with objects containing 'name', 'description', and 'prompt' fields. 
The attributes should be diverse and capture different aspects relevant to the goal.
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in designing evaluation frameworks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        attributes = result.get("attributes", [])
        
        # Cache the results
        cache[cache_key] = attributes
        save_cache(cache, "attributes")
        
        return attributes
        
    except Exception as e:
        st.error(f"Error generating attributes: {e}")
        # Fallback to default attributes
        return [
            {
                "name": "relevance",
                "description": "How relevant the item is to the goal",
                "prompt": "How much more relevant to the goal is item A than item B?"
            },
            {
                "name": "impact",
                "description": "The potential impact of the item",
                "prompt": "How much more impactful is item A than item B?"
            },
            {
                "name": "quality",
                "description": "The overall quality of the item",
                "prompt": "How much higher quality is item A than item B?"
            }
        ]

def query_llm_for_attribute_comparisons(attributes, goal):
    """
    Query the LLM to do pairwise comparisons of attributes
    Returns a pairwise comparison matrix for the attributes
    """
    cache = load_cache("attribute_weights")
    attributes_key = "+".join([attr["name"] for attr in attributes]) + "_" + goal
    
    if attributes_key in cache:
        st.success("Using cached attribute weights")
        return cache[attributes_key]
    
    st.info("Comparing attributes based on goal...")
    
    n = len(attributes)
    pcm = np.ones((n, n))
    
    try:
        # For each pair of attributes, ask LLM to compare
        for i in range(n):
            for j in range(i+1, n):
                attr_a = attributes[i]
                attr_b = attributes[j]
                
                prompt = f"""In the context of this goal: "{goal}"

I need to compare the relative importance of two attributes:

Attribute A: {attr_a['name']} - {attr_a['description']}
Attribute B: {attr_b['name']} - {attr_b['description']}

How much more important is Attribute A compared to Attribute B for achieving the goal?
Express your answer as a single ratio number.
- If A is twice as important as B, respond with "2"
- If A is slightly more important than B, respond with a value between 1 and 2 (e.g., "1.5")
- If A and B are equally important, respond with "1"
- If B is more important than A, respond with a value less than 1 (e.g., "0.5" if B is twice as important as A)

Provide only the numerical value with no additional text."""

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant skilled in analytical decision making."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=10
                )
                
                try:
                    ratio = float(response.choices[0].message.content.strip())
                    pcm[i, j] = ratio
                    pcm[j, i] = 1.0 / ratio
                except ValueError:
                    st.warning(f"Could not parse response for {attr_a['name']} vs {attr_b['name']}, using default value of 1")
                    pcm[i, j] = 1.0
                    pcm[j, i] = 1.0
        
        # Cache the results
        cache[attributes_key] = pcm.tolist()
        save_cache(cache, "attribute_weights")
        
        return pcm.tolist()
        
    except Exception as e:
        st.error(f"Error comparing attributes: {e}")
        # Return identity matrix as fallback
        return np.ones((n, n)).tolist()

def query_llm_for_item_comparisons(items, attribute):
    """
    Query the LLM to do pairwise comparisons of items for a specific attribute
    Returns a pairwise comparison matrix for the items
    """
    cache = load_cache("item_comparisons")
    cache_key = attribute["name"]
    
    if cache_key in cache and len(cache[cache_key].get("item_pairs", {})) >= (len(items) * (len(items) - 1)):
        st.success(f"Using cached comparisons for {attribute['name']}")
        
        # Reconstruct the matrix from cache
        n = len(items)
        pcm = np.ones((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                pair_key = get_cache_key(i, j)
                if pair_key in cache[cache_key]["item_pairs"]:
                    ratio = cache[cache_key]["item_pairs"][pair_key]
                    pcm[i, j] = ratio
                    pcm[j, i] = 1.0 / ratio if ratio != 0 else 0
        
        return pcm.tolist()
    
    # Initialize cache structure for this attribute if it doesn't exist
    if cache_key not in cache:
        cache[cache_key] = {"item_pairs": {}}
    
    st.info(f"Comparing items based on {attribute['name']}...")
    
    n = len(items)
    pcm = np.ones((n, n))
    
    # Calculate total comparisons
    total_comparisons = n * (n - 1) // 2
    progress_bar = st.progress(0)
    status_text = st.empty()
    comparisons_done = 0
    
    try:
        # For each pair of items, ask LLM to compare
        for i in range(n):
            for j in range(i+1, n):
                item_a = items[i]
                item_b = items[j]
                
                # Check if comparison is already cached
                pair_key = get_cache_key(i, j)
                if pair_key in cache[cache_key]["item_pairs"]:
                    ratio = cache[cache_key]["item_pairs"][pair_key]
                    status_text.text(f"Using cached comparison for Item {i+1} with Item {j+1}")
                else:
                    status_text.text(f"Comparing Item {i+1} with Item {j+1} for {attribute['name']}...")
                    
                    prompt = f"""{attribute['prompt']}

Item A: {item_a}

Item B: {item_b}

Respond with a single numerical ratio. For example, if A is twice as good as B, respond with "2". 
If A is half as good as B, respond with "0.5". If they're equal, respond with "1"."""

                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that evaluates relative properties of items."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=10
                    )
                    
                    try:
                        ratio = float(response.choices[0].message.content.strip())
                        # Cache the result
                        cache[cache_key]["item_pairs"][pair_key] = ratio
                        cache[cache_key]["item_pairs"][get_cache_key(j, i)] = 1.0 / ratio if ratio != 0 else 0
                        save_cache(cache, "item_comparisons")
                    except ValueError:
                        st.warning(f"Could not parse response for comparison of items {i+1} and {j+1}, using default value of 1")
                        ratio = 1.0
                
                pcm[i, j] = ratio
                pcm[j, i] = 1.0 / ratio if ratio != 0 else 0
                
                # Update progress
                comparisons_done += 1
                progress_bar.progress(comparisons_done / total_comparisons)
        
        # Clear status text and progress bar when done
        status_text.empty()
        progress_bar.empty()
        
        return pcm.tolist()
        
    except Exception as e:
        st.error(f"Error comparing items for {attribute['name']}: {e}")
        # Return identity matrix as fallback
        return np.ones((n, n)).tolist()

def calculate_ahp_scores(items, attributes, attribute_weights, item_matrices):
    """
    Calculate the final AHP scores for each item
    """
    n_items = len(items)
    n_attrs = len(attributes)
    
    # Calculate the weight for each attribute
    attr_weights = solve_pcm_geometric_mean(np.array(attribute_weights))
    
    # Calculate item scores for each attribute
    item_scores = np.zeros((n_attrs, n_items))
    for i, attr in enumerate(attributes):
        # Get the item matrix for this attribute
        item_matrix = item_matrices[attr["name"]]
        # Calculate weights using geometric mean
        item_scores[i] = solve_pcm_geometric_mean(np.array(item_matrix))
    
    # Final scores are weighted sum of attribute scores
    final_scores = np.zeros(n_items)
    for i in range(n_items):
        for j in range(n_attrs):
            final_scores[i] += attr_weights[j] * item_scores[j, i]
    
    # Return normalized scores (sum to 1)
    return final_scores / np.sum(final_scores)

def query_llm_for_direct_ranking(items, goal):
    """
    Query the LLM to directly rank the items based on the goal
    """
    cache = load_cache("direct_ranking")
    items_hash = str(hash(str(items) + goal))
    
    if items_hash in cache:
        st.success("Using cached direct ranking")
        return cache[items_hash]
    
    st.info("Getting direct ranking of items...")
    
    try:
        prompt = f"""Please directly rank the following items based on this goal: "{goal}"

Items to rank:
{json.dumps(items, indent=2)}

For each item, assign a utility score from 0 to 100, where 100 is the highest possible utility. 
Think step by step about how each item contributes to the goal.

Format your response as a JSON object with the following structure:
{{
  "scores": {{
    "0": 85,
    "1": 62,
    ...
  }},
  "reasoning": "Detailed explanation of your reasoning, especially for the top 3 ranked items."
}}

Where the keys in the "scores" object are the item indices (starting from 0).
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in analytical decision making."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract scores and reasoning
        scores = {}
        reasoning = ""
        
        # Handle different possible JSON structures
        if "scores" in result:
            scores = result["scores"]
        elif isinstance(result, dict) and all(k.isdigit() for k in result.keys()):
            # Handle case where the model returned just the scores object directly
            scores = result
        
        if "reasoning" in result:
            reasoning = result["reasoning"]
        
        # Convert string keys to integers if needed
        scores_dict = {int(k): float(v) for k, v in scores.items()}
        
        # Normalize scores to sum to 1
        total = sum(scores_dict.values())
        if total > 0:  # Prevent division by zero
            normalized_scores = {k: v/total for k, v in scores_dict.items()}
        else:
            normalized_scores = scores_dict
        
        # Cache the results
        cache[items_hash] = {"scores": normalized_scores, "reasoning": reasoning}
        save_cache(cache, "direct_ranking")
        
        return {"scores": normalized_scores, "reasoning": reasoning}
        
    except Exception as e:
        st.error(f"Error getting direct ranking: {e}")
        # Return empty dict as fallback
        return {"scores": {}, "reasoning": ""}

def run_ahp_analysis():
    """
    Run the complete AHP analysis process
    """
    items = st.session_state.items
    goal = st.session_state.goal
    
    if not items:
        st.error("Please add some items to evaluate")
        return
    
    with st.spinner("Running AHP analysis..."):
        # Step 1: Generate attributes
        attributes = query_llm_for_attributes(items, goal)
        st.session_state.attributes = attributes
        
        # Step 2: Compare attributes
        attribute_matrix = query_llm_for_attribute_comparisons(attributes, goal)
        st.session_state.attribute_weights = attribute_matrix
        
        # Step 3: For each attribute, compare items
        item_matrices = {}
        for attr in attributes:
            item_matrix = query_llm_for_item_comparisons(items, attr)
            item_matrices[attr["name"]] = item_matrix
        
        st.session_state.attribute_matrices = item_matrices
        
        # Step 4: Calculate AHP scores
        ahp_scores = calculate_ahp_scores(items, attributes, attribute_matrix, item_matrices)
        st.session_state.ahp_scores = {i: score for i, score in enumerate(ahp_scores)}
        
        # Step 5: Get direct ranking for comparison
        direct_result = query_llm_for_direct_ranking(items, goal)
        st.session_state.direct_scores = direct_result["scores"]
        st.session_state.direct_reasoning = direct_result.get("reasoning", "")
    
    st.success("AHP analysis completed!")

# Main app UI
st.title("Analytic Hierarchy Process (AHP) Agent")
st.write("This app uses the Analytic Hierarchy Process to evaluate and rank items based on multiple criteria.")

# Setup section
st.header("Setup")

# Input the goal
st.subheader("Goal")
goal_input = st.text_area("What is your evaluation goal?", value=st.session_state.goal, height=100)
if goal_input != st.session_state.goal:
    st.session_state.goal = goal_input

# Input items to evaluate
st.subheader("Items to Evaluate")

# Create two columns for the items section
col1, col2 = st.columns([3, 1])

with col1:
    item_count = len(st.session_state.items)
    item_text = "\n\n".join(st.session_state.items)
    
    items_input = st.text_area(
        "Enter the items to evaluate (one per line or separated by blank lines):",
        value=item_text,
        height=300
    )
    
    # Parse and update items when input changes
    if items_input != item_text:
        # Split by double newlines or single newlines
        new_items = re.split(r'\n\n|\n', items_input)
        # Filter out empty items
        new_items = [item.strip() for item in new_items if item.strip()]
        st.session_state.items = new_items

with col2:
    st.write("**Add Individual Items**")
    
    new_item = st.text_area("New item:", height=100, key="new_item_input")
    
    if st.button("Add Item", use_container_width=True):
        if new_item.strip():
            st.session_state.items.append(new_item.strip())
            # Clear the input
            st.session_state.new_item_input = ""
            st.rerun()
    
    if st.button("Clear All Items", use_container_width=True):
        st.session_state.items = []
        st.rerun()
    
    st.write(f"Total items: **{len(st.session_state.items)}**")

# Run analysis button
if st.button("Run AHP Analysis", type="primary", use_container_width=True):
    run_ahp_analysis()

st.markdown("---")

# Main area for results
if st.session_state.attributes:
    # Display the attributes
    st.header("Attributes for Evaluation")
    
    attributes_df = pd.DataFrame([
        {
            "Name": attr["name"], 
            "Description": attr.get("description", ""), 
            "Comparison Prompt": attr["prompt"]
        } 
        for attr in st.session_state.attributes
    ])
    
    st.dataframe(attributes_df, hide_index=True)
    
    # Display attribute weights
    if st.session_state.attribute_weights:
        st.header("Attribute Importance Weights")
        
        # Calculate attribute weights
        attr_weights = solve_pcm_geometric_mean(np.array(st.session_state.attribute_weights))
        
        # Create a DataFrame for attribute weights
        attr_weight_df = pd.DataFrame({
            "Attribute": [attr["name"] for attr in st.session_state.attributes],
            "Weight": attr_weights
        })
        
        # Sort by weight descending
        attr_weight_df = attr_weight_df.sort_values("Weight", ascending=False)
        
        # Display as a bar chart
        st.bar_chart(attr_weight_df.set_index("Attribute"))
        
        # Display the pairwise comparison matrix
        with st.expander("View Attribute Pairwise Comparison Matrix"):
            attr_names = [attr["name"] for attr in st.session_state.attributes]
            attr_pcm_df = pd.DataFrame(st.session_state.attribute_weights, index=attr_names, columns=attr_names)
            st.dataframe(attr_pcm_df.style.format("{:.2f}"))
    
    # Display item comparison matrices for each attribute
    if st.session_state.attribute_matrices:
        st.header("Item Comparisons by Attribute")
        
        # Create tabs for each attribute
        tabs = st.tabs([attr["name"] for attr in st.session_state.attributes])
        
        for i, attr in enumerate(st.session_state.attributes):
            with tabs[i]:
                # Get the matrix for this attribute
                matrix = st.session_state.attribute_matrices[attr["name"]]
                
                # Calculate weights for this attribute
                weights = solve_pcm_geometric_mean(np.array(matrix))
                
                # Display the weights as a bar chart
                item_weights_df = pd.DataFrame({
                    "Item": [f"Item {j+1}" for j in range(len(st.session_state.items))],
                    "Weight": weights
                })
                
                st.subheader(f"Weights for {attr['name']}")
                st.bar_chart(item_weights_df.set_index("Item"))
                
                # Display the pairwise comparison matrix
                with st.expander(f"View Pairwise Comparison Matrix for {attr['name']}"):
                    item_labels = [f"Item {j+1}" for j in range(len(st.session_state.items))]
                    pcm_df = pd.DataFrame(matrix, index=item_labels, columns=item_labels)
                    st.dataframe(pcm_df.style.format("{:.2f}"))
    
    # Display final AHP scores
    if st.session_state.ahp_scores:
        st.header("AHP Results")
        
        # Format scores for display
        items = st.session_state.items
        scores = st.session_state.ahp_scores
        
        # Create a DataFrame with items and scores
        results_df = pd.DataFrame({
            "Item #": [f"Item {i+1}" for i in range(len(items))],
            "Item Text": [item[:100] + "..." if len(item) > 100 else item for item in items],
            "AHP Score": [scores[i] for i in range(len(items))]
        })
        
        # Sort by score descending
        results_df = results_df.sort_values("AHP Score", ascending=False)
        
        # Display as a table
        st.dataframe(results_df.style.format({"AHP Score": "{:.4f}"}), hide_index=True)
        
        # Display as a bar chart
        chart_df = pd.DataFrame({
            "Item": [f"Item {i+1}" for i in range(len(items))],
            "Score": [scores[i] for i in range(len(items))]
        })
        chart_df = chart_df.sort_values("Score", ascending=False)
        
        st.bar_chart(chart_df.set_index("Item"))
        
        # Display the top ranked items with full text
        st.subheader("Top Ranked Items")
        
        # Get the indices of the top 3 items (or fewer if there aren't 3)
        top_count = min(3, len(items))
        
        # Get the indices sorted by score
        sorted_indices = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        
        # Display the top items
        for rank, idx in enumerate(sorted_indices[:top_count]):
            with st.container(border=True):
                st.markdown(f"#### {rank+1}. Item {idx+1} (Score: {scores[idx]:.4f})")
                st.markdown(f"_{items[idx]}_")
        
        # Show all items in an expander
        with st.expander("View All Items Ranked"):
            for rank, idx in enumerate(sorted_indices):
                st.markdown(f"**{rank+1}. Item {idx+1} (Score: {scores[idx]:.4f})**")
                st.markdown(f"{items[idx]}")
                st.markdown("---")
    
    # Compare AHP with direct ranking
    if st.session_state.ahp_scores and st.session_state.direct_scores:
        st.header("AHP vs Direct Ranking Comparison")
        
        items = st.session_state.items
        ahp_scores = st.session_state.ahp_scores
        direct_scores = st.session_state.direct_scores
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            "Item #": [f"Item {i+1}" for i in range(len(items))],
            "Item Text": [item[:100] + "..." if len(item) > 100 else item for item in items],
            "AHP Score": [ahp_scores.get(i, 0) for i in range(len(items))],
            "Direct Score": [direct_scores.get(i, 0) for i in range(len(items))]
        })
        
        # Calculate difference and rank difference
        comparison_df["Difference"] = comparison_df["AHP Score"] - comparison_df["Direct Score"]
        
        # Sort by AHP score
        comparison_df = comparison_df.sort_values("AHP Score", ascending=False)
        
        # Display as a table
        st.dataframe(comparison_df.style.format({
            "AHP Score": "{:.4f}",
            "Direct Score": "{:.4f}",
            "Difference": "{:.4f}"
        }), hide_index=True)
        
        # Display side-by-side top items
        st.subheader("Top Items Comparison")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top 3 by AHP")
            # Get indices sorted by AHP score
            ahp_sorted = sorted(range(len(items)), key=lambda i: ahp_scores.get(i, 0), reverse=True)
            
            # Display top 3 AHP items
            for rank, idx in enumerate(ahp_sorted[:3]):
                with st.container(border=True):
                    st.markdown(f"**{rank+1}. Item {idx+1} (Score: {ahp_scores.get(idx, 0):.4f})**")
                    st.markdown(f"{items[idx]}")
        
        with col2:
            st.markdown("### Top 3 by Direct Ranking")
            # Get indices sorted by direct score
            direct_sorted = sorted(range(len(items)), key=lambda i: direct_scores.get(i, 0), reverse=True)
            
            # Display top 3 direct items
            for rank, idx in enumerate(direct_sorted[:3]):
                with st.container(border=True):
                    st.markdown(f"**{rank+1}. Item {idx+1} (Score: {direct_scores.get(idx, 0):.4f})**")
                    st.markdown(f"{items[idx]}")
        
        # Display reasoning for direct ranking
        if hasattr(st.session_state, 'direct_reasoning') and st.session_state.direct_reasoning:
            with st.expander("Direct Ranking Reasoning"):
                st.write(st.session_state.direct_reasoning)
        
        # Display as a comparison chart
        st.subheader("Score Comparison Chart")
        chart_data = pd.DataFrame({
            "Item": [f"Item {i+1}" for i in range(len(items))],
            "AHP": [ahp_scores.get(i, 0) for i in range(len(items))],
            "Direct": [direct_scores.get(i, 0) for i in range(len(items))]
        })
        
        # Sort by AHP score
        chart_data = chart_data.sort_values("AHP", ascending=False)
        
        # Reshape for charting
        chart_data_melted = pd.melt(
            chart_data, 
            id_vars=["Item"], 
            value_vars=["AHP", "Direct"],
            var_name="Method", 
            value_name="Score"
        )
        
        # Create a grouped bar chart
        st.bar_chart(chart_data_melted.pivot(index="Item", columns="Method", values="Score"))
        
        # Calculate correlation and agreement metrics
        correlation = np.corrcoef(
            [ahp_scores.get(i, 0) for i in range(len(items))],
            [direct_scores.get(i, 0) for i in range(len(items))]
        )[0, 1]
        
        # Calculate rank correlation (Spearman)
        # First get the rankings (not the scores)
        ahp_ranks = {idx: rank for rank, idx in enumerate(ahp_sorted)}
        direct_ranks = {idx: rank for rank, idx in enumerate(direct_sorted)}
        
        # Calculate the sum of squared differences in ranks
        d_squared_sum = sum((ahp_ranks.get(i, 0) - direct_ranks.get(i, 0))**2 for i in range(len(items)))
        
        # Calculate Spearman's rank correlation coefficient
        n = len(items)
        if n > 1:
            spearman = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        else:
            spearman = 1.0
        
        # Calculate agreement on top items
        top_k = min(3, len(items))
        top_ahp = set(ahp_sorted[:top_k])
        top_direct = set(direct_sorted[:top_k])
        agreement = len(top_ahp.intersection(top_direct)) / top_k
        
        # Display correlation metrics
        st.subheader("Agreement Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Pearson Correlation", f"{correlation:.3f}", 
                     help="Pearson correlation between AHP and Direct scores (1.0 = perfect correlation)")
        
        with metrics_col2:
            st.metric("Rank Correlation", f"{spearman:.3f}", 
                     help="Spearman's rank correlation between AHP and Direct rankings (1.0 = identical ranking order)")
        
        with metrics_col3:
            st.metric(f"Top {top_k} Agreement", f"{agreement:.0%}", 
                     help=f"Percentage of items that appear in both top {top_k} lists")
else:
    st.info("Click 'Run AHP Analysis' to start the process.")

# Footer
st.markdown("---")
st.caption("Analytic Hierarchy Process (AHP) Agent - Uses LLM to decompose complex evaluation into manageable pairwise comparisons")