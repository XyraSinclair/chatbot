import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
openai_api_key = st.secrets.openai_api_key
client = OpenAI(api_key=openai_api_key)

tweets = [
    r"i just muted 500ppl, cleansed following, purged interests in settings, added a bunch of block words, searched for things i find interesting and liked all of it, and my feed went from 97% politics and shit posts back to old x in an hour",
    r"Survival mode is not just feeling stressed—it's your body operating on autopilot to protect you.",
    r"t's more important to develop a values system that is broadly applicable than worrying about the specific application. i would be scared to ever figure my life out too much, all the cool stuff has happened in the liminal spaces.",
    r"The Jhanas are real. This is a crazy fact. All the things they say about bliss on demand, infinite blah blah, they're real. Hard to put to words obviously, but the closest phrase I have to convey the feeling is \"infinite happiness\" (on J3). ",
    r"The more you ignore your needs, the more they control you.",
    r"Not a lot of people understand this... but you actually don't have to have an opinion about everything. You don't have to decide if something is good or bad. Marcus Aurelius says limiting the amount of opinions we have is one of the most powerful things we can do in life.",
    r"Healing is not meant to leave us in a spot where the only skill in our arsenal is unpacking & analyzing our emotions. The idea is to develop that skill *in service of* becoming more flexible, courageous and capable of adapting to the lives we actually want.",
    r"I tentatively believe that people who want to \"get into meditation\" are probably better suited cleaning up diet and lifestyle first however they can, and also doing a ton of forgiveness meditation before worrying about anything like shamatha or vipassana. clearing out the system",
    r"Life is going so fast because you're not injecting enough silence into your days. Spend an hour reading or writing or walking in silence and watch how your day seems to double in duration",
    r"last night i dreamt i was healing men and helping even the toughest ones with the hardest exteriors relax into their softness and remember their deep, embodied, present, alive nature within , showing them they don't need to always Prove or Fight . that sometimes they can Just Be.",
    r"mental tension, a subtle form of internal violence faster than thought, is at the root of all your problems",
    r"i'm not sure why people are so afraid to be seen when all the best things in life happen when you expose yourself more courageously to the world",
    r"it is essential to see your intuition as your anchor instead of other people's opinions of you as your anchor",
    r"reminder: you definitely don't need any expensive biotech to stimulate your vagus nerve and self-regulate. Low-pitch humming works absurdly well."
]

attributes = [
    ("practicality", "how much more practical/advice-giving is tweet A than tweet B?"),
    ("wisdom", "how much more wisdom-based or philosophical is tweet A than tweet B?"),
    ("meditativeness", "how much more about self-regulation, presence, or awareness is tweet A than tweet B?"),
    ("perspective_shift", "how much more cognitively liberating or perspective-shifting is tweet A than tweet B?"),
    ("memorability", "how much more likely is tweet A to be remembered after 24 hours than tweet B?"),
    ("stickiness", "how much more likely is tweet A to change someone's behavior long-term than tweet B?"),
    ("emotion_intensity", "how much more emotionally powerful is tweet A than tweet B?"),
    ("actionability", "how much more immediately actionable is tweet A than tweet B?"),
    ("spirituality", "how much more does tweet A touch on deep spiritual or existential themes than tweet B?"),
    ("groundedness", "how much more grounded and down-to-earth is tweet A than tweet B?"),
    ("control_vs_surrender",
     "how much more does tweet A emphasize control over one's life rather than surrendering to its flow?"),
    ("mystical_energy",
     "how much more mysterious, surreal, or dreamlike is tweet A than tweet B?"),
    ("calming_effect", "how much more likely is tweet A to make someone feel calm and reassured than tweet B?"),
    ("self_inquiry", "how much more does tweet A encourage deep personal reflection than tweet B?"),
    ("truth_bomb", "how much more does tweet A feel like a direct, undeniable truth bomb than tweet B?")
]

attribute_scores = {}

def analyze_pairwise_comparisons():
    for attribute_name, attribute_prompt in attributes:
        if attribute_name not in attribute_scores:
            attribute_scores[attribute_name] = {}
            
        comparison_matrix = []
        n = len(tweets)
        
        # Create empty n×n matrix
        for i in range(n):
            comparison_matrix.append([0] * n)
            # Set diagonal to 1 (self-comparison)
            comparison_matrix[i][i] = 1
        
        # Fill the upper triangular part of the matrix
        for i in range(n):
            for j in range(i+1, n):
                tweet_a = tweets[i]
                tweet_b = tweets[j]
                
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
                    # If response is not a clean number, default to 1
                    ratio = 1
                
                # Fill both positions in the matrix
                comparison_matrix[i][j] = ratio
                comparison_matrix[j][i] = 1 / ratio if ratio != 0 else 0
        
        attribute_scores[attribute_name] = comparison_matrix
    
    return attribute_scores

# Initialize the app with data on button press
if st.button("Run Pairwise Comparisons"):
    with st.spinner("Processing pairwise comparisons..."):
        analyze_pairwise_comparisons()


# Show title and description.
st.title("How do LLMs perform at pairwise ratio comparison?")
st.write("This app demonstrates how LLMs can be used to create complete pairwise ratio comparison matrices for tweet analysis.")
st.write("Select an attribute and run the comparisons to see how different tweets compare on that dimension.")

# Add a numerical ID to each tweet for easier display
numbered_tweets = {f"Tweet {i+1}": tweet for i, tweet in enumerate(tweets)}

# Display tweets for reference
with st.expander("View All Tweets"):
    for tweet_id, tweet_text in numbered_tweets.items():
        st.markdown(f"**{tweet_id}:** {tweet_text}")

# Select attribute to display
selected_attribute = st.selectbox("Select attribute to analyze:", 
                                [attr[0] for attr in attributes])

# Display results if available
if attribute_scores and selected_attribute in attribute_scores:
    st.subheader(f"Pairwise Comparison Matrix for '{selected_attribute}'")
    
    # Get the matrix for the selected attribute
    matrix = attribute_scores[selected_attribute]
    
    # Display as a table
    import numpy as np
    import pandas as pd
    
    # Create labels for the matrix
    labels = [f"Tweet {i+1}" for i in range(len(tweets))]
    
    # Convert to pandas DataFrame for better display
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    
    # Display the matrix
    st.dataframe(df.style.format("{:.2f}"))
    
    # Calculate and display consistency ratio if matrix is complete
    is_complete = all(all(cell != 0 for cell in row) for row in matrix)
    if is_complete:
        # Calculate consistency measures
        import numpy as np
        matrix_np = np.array(matrix)
        
        # Principal eigenvector calculation (using power method)
        n = len(matrix_np)
        eigenvector = np.ones(n)
        for _ in range(20):  # 20 iterations is usually sufficient
            eigenvector = matrix_np @ eigenvector
            eigenvector = eigenvector / np.linalg.norm(eigenvector)
        
        # Normalize to get weights
        weights = eigenvector / np.sum(eigenvector)
        
        # Display weights
        st.subheader("Feature Weights (from Eigenvector)")
        weight_df = pd.DataFrame({
            'Tweet': [f"Tweet {i+1}" for i in range(n)],
            'Weight': weights
        })
        st.dataframe(weight_df.sort_values('Weight', ascending=False).style.format({'Weight': '{:.4f}'}))
        
        # Display tweets in ranked order
        st.subheader("Tweets Ranked by " + selected_attribute.title())
        sorted_indices = np.argsort(-weights)  # Negative to sort descending
        for rank, idx in enumerate(sorted_indices):
            st.markdown(f"**Rank {rank+1} (Weight: {weights[idx]:.4f}):** {tweets[idx]}")
            
    else:
        st.warning("Matrix is incomplete. Some comparisons are missing.")
else:
    st.info("Run the pairwise comparisons to generate results.")

# Add a section for asking questions about the results
st.subheader("Ask Questions About the Analysis")
st.write("You can ask questions about the pairwise comparison results or general questions about the methodology.")

# Create a session state variable to store the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field
if prompt := st.chat_input("Ask a question about the pairwise comparisons..."):
    # Store and display the current prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Create context about the current state of analysis
    context = "The app analyzes tweets using pairwise ratio comparisons for various attributes like practicality, wisdom, etc."
    
    if attribute_scores:
        # Add information about which attributes have been analyzed
        analyzed_attributes = list(attribute_scores.keys())
        context += f"\n\nAttributes that have been analyzed: {', '.join(analyzed_attributes)}"
        
        # If a specific attribute is selected, add details about it
        if selected_attribute in attribute_scores:
            context += f"\n\nCurrently viewing: {selected_attribute}"
            
            # Add top ranked tweets for this attribute if available
            matrix = attribute_scores[selected_attribute]
            is_complete = all(all(cell != 0 for cell in row) for row in matrix)
            
            if is_complete:
                matrix_np = np.array(matrix)
                n = len(matrix_np)
                eigenvector = np.ones(n)
                for _ in range(20):
                    eigenvector = matrix_np @ eigenvector
                    eigenvector = eigenvector / np.linalg.norm(eigenvector)
                weights = eigenvector / np.sum(eigenvector)
                
                sorted_indices = np.argsort(-weights)
                top_tweets = [f"Tweet {idx+1}" for idx in sorted_indices[:3]]
                context += f"\n\nTop 3 tweets for {selected_attribute}: {', '.join(top_tweets)}"
    
    # Generate a response using the OpenAI API
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are an assistant that helps explain pairwise ratio comparison analysis. Here's the current context of the application: {context}"},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ],
        stream=True,
    )

    # Stream the response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
