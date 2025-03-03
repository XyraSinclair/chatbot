import streamlit as st
from openai import OpenAI

tweets = [
    r"i just muted 500ppl, cleansed following, purged interests in settings, added a bunch of block words, searched for things i find interesting and liked all of it, and my feed went from 97% politics and shit posts back to old x in an hour",
    r"Survival mode is not just feeling stressed—it’s your body operating on autopilot to protect you.",
    r"t’s more important to develop a values system that is broadly applicable than worrying about the specific application. i would be scared to ever figure my life out too much, all the cool stuff has happened in the liminal spaces.",
    r"The Jhanas are real. This is a crazy fact. All the things they say about bliss on demand, infinite blah blah, they're real. Hard to put to words obviously, but the closest phrase I have to convey the feeling is \"infinite happiness\" (on J3). ",
    r"The more you ignore your needs, the more they control you.",
    r"Not a lot of people understand this... but you actually don’t have to have an opinion about everything. You don’t have to decide if something is good or bad. Marcus Aurelius says limiting the amount of opinions we have is one of the most powerful things we can do in life.",
    r"Healing is not meant to leave us in a spot where the only skill in our arsenal is unpacking & analyzing our emotions. The idea is to develop that skill *in service of* becoming more flexible, courageous and capable of adapting to the lives we actually want.",
    r"I tentatively believe that people who want to “get into meditation” are probably better suited cleaning up diet and lifestyle first however they can, and also doing a ton of forgiveness meditation before worrying about anything like shamatha or vipassana. clearing out the system",
    r"Life is going so fast because you're not injecting enough silence into your days. Spend an hour reading or writing or walking in silence and watch how your day seems to double in duration",
    r"last night i dreamt i was healing men and helping even the toughest ones with the hardest exteriors relax into their softness and remember their deep, embodied, present, alive nature within , showing them they don’t need to always Prove or Fight . that sometimes they can Just Be.",
    r"mental tension, a subtle form of internal violence faster than thought, is at the root of all your problems",
    r"i’m not sure why people are so afraid to be seen when all the best things in life happen when you expose yourself more courageously to the world",
    r"it is essential to see your intuition as your anchor instead of other people’s opinions of you as your anchor",
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
    ("control_vs_surrender", "how much more does tweet A emphasize control over one's life rather than surrendering to its flow?"),
    ("mystical_energy", "how much more mysterious, surreal, or dreamlike is tweet A than tweet B?"),
    ("calming_effect", "how much more likely is tweet A to make someone feel calm and reassured than tweet B?"),
    ("self_inquiry", "how much more does tweet A encourage deep personal reflection than tweet B?"),
    ("truth_bomb", "how much more does tweet A feel like a direct, undeniable truth bomb than tweet B?")
]

attribute_scores = []

for attribute in attributes:
    pairwise_comparison_matrix = []
    for entity in entities:
        

# Show title and description.
st.title("How do LLMs perform at pairwise ratio comparison?")
st.write(
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
    "How well do LLMs rate tweets?"
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.openai_api_key


# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API.
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    )

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
