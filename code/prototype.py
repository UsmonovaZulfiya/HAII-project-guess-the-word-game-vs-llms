import os
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

os.environ["NEBIUS_API_KEY"] = "v1.CmMKHHN0YXRpY2tleS1lMDByNG13OHJlOTEwYXhtZjcSIXNlcnZpY2VhY2NvdW50LWUwMHZ2ZW53eDUwMDM1NTU1NjIMCOeCosgGEPu9g74COgsI54W6kwcQwMLuSkACWgNlMDA.AAAAAAAAAAF-s3IVuPd-6SwZfzos0vgqlAlUZtfge6Kj5JAVVepABWajqetR76LusvMMN1mo0E5Y5TbLdhzBkjxNaiMXxrQM"
MODEL_NAME = "google/gemma-3-27b-it"

# CONCURRENCY SETTINGS
# Increase this if the API is fast. Decrease if you get "Rate Limit" errors.
MAX_WORKERS = 20

def build_model():
    return ChatOpenAI(
        base_url="https://api.studio.nebius.ai/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
        model=MODEL_NAME,
        temperature=0.7,
        max_retries=3, # Increased retries for stability in parallel mode
        request_timeout=30
    )

# We create a thread-local model builder or just rely on LangChain's thread safety.
# For simplicity, we will instantiate the model inside the workers or pass it carefully.
# LangChain clients are generally thread-safe.
global_model = build_model()

# ==========================================
# 2. EXPERIMENT PARAMETERS
# ==========================================

WORD_LIST_1 = [
    "Apple", "Baby", "Ball", "Banana", "Bed", "Bird", "Boat", "Book",
    "Bottle", "Box", "Boy", "Bread", "Bus", "Cake", "Camera", "Car",
    "Cat", "Chair", "Chicken", "Child", "Clock", "Cloud", "Coat",
    "Coffee", "Computer", "Corn", "Cow", "Cup", "Desk", "Doctor",
    "Dog", "Door", "Dress", "Ear", "Egg", "Eye", "Face", "Farm",
    "Fire", "Fish", "Floor", "Flower", "Food", "Foot", "Fork",
    "Friend", "Fruit", "Garden", "Girl", "Glass", "Gold", "Grass",
    "Hair", "Hand", "Hat", "Head", "Heart", "Home", "Horse", "House",
    "Ice", "Key", "Knife", "Lamp", "Leaf", "Leg", "Letter", "Light",
    "Man", "Map", "Meat", "Milk", "Money", "Moon", "Morning",
    "Mother", "Mountain", "Mouse", "Mouth", "Music", "Night", "Nose",
    "Ocean", "Office", "Oil", "Paper", "Park", "Pen", "Phone",
    "Picture", "Pig", "Pizza", "Plane", "Plant", "Plate", "Rain",
    "Ring", "River", "Road", "Rock"
]

WORD_LIST_2 = [
    "Abacus", "Accordion", "Acorn", "Anvil", "Archipelago", "Armadillo",
    "Artichoke", "Asparagus", "Awl", "Bagpipe", "Banjo", "Barnacle",
    "Barometer", "Basilisk", "Bassoon", "Baton", "Beaker", "Beaver",
    "Bellows", "Beret", "Bifocals", "Biscuit", "Blimp", "Bonnet",
    "Boomerang", "Bouquet", "Bramble", "Brocade", "Buckle", "Bungalow",
    "Cactus", "Cauldron", "Cello", "Centipede", "Chalice", "Chandelier",
    "Chariot", "Chisel", "Chrysanthemum", "Clarinet", "Cleaver", "Cobweb",
    "Cockroach", "Colander", "Compass", "Corset", "Coyote", "Crayfish",
    "Crowbar", "Crucible", "Cuckoo", "Curio", "Cutlass", "Cymbal",
    "Dagger", "Dandelion", "Decanter", "Dirigible", "Dodo", "Dragonfly",
    "Drawbridge", "Dumbbell", "Dynamite", "Easel", "Eclair", "Emus",
    "Epaulet", "Ermine", "Espadrille", "Falcon", "Fedora", "Fez",
    "Fiddle", "Figurine", "Flute", "Fresco", "Fuselage", "Gable",
    "Gadget", "Galleon", "Gargoyle", "Gauntlet", "Gazebo", "Geode",
    "Gerbil", "Geyser", "Gimlet", "Glacier", "Glider", "Gnome",
    "Goblet", "Gondola", "Gong", "Gopher", "Gorilla"
]

# WORD_LIST_3 = [
#     "Ability", "Action", "Advice", "Age", "Agreement", "Air", "Anger",
#     "Answer", "Art", "Attention", "Authority", "Beauty", "Belief",
#     "Benefit", "Business", "Care", "Case", "Cause", "Chance", "Change",
#     "Choice", "City", "Class", "Color", "Comfort", "Company", "Condition",
#     "Control", "Cost", "Country", "Course", "Credit", "Culture", "Danger",
#     "Data", "Date", "Death", "Decision", "Degree", "Design", "Desire",
#     "Detail", "Difference", "Direction", "Discussion", "Disease", "Doubt",
#     "Dream", "Duty", "Economy", "Effect", "Effort", "Energy", "Error",
#     "Event", "Example", "Experience", "Fact", "Failure", "Faith", "Family",
#     "Fear", "Feeling", "Force", "Form", "Freedom", "Friendship", "Fun",
#     "Future", "Game", "Goal", "Government", "Group", "Growth", "Happiness",
#     "Hate", "Health", "Help", "History", "Hope", "Idea", "Image", "Impact",
#     "Industry", "Information", "Interest", "Issue", "Job", "Joy",
#     "Justice", "Knowledge", "Law", "Level", "Life", "Line", "Love", "Luck",
#     "Market", "Matter", "Meaning"
# ]

# WORD_LIST_4 = [
#     "Aberration", "Acrimony", "Adulation", "Affluence", "Alacrity",
#     "Allegory", "Ambivalence", "Amnesty", "Anarchy", "Animosity",
#     "Anomaly", "Apathy", "Apex", "Aptitude", "Arrogance", "Atonement",
#     "Atrophy", "Audacity", "Avarice", "Aversion", "Banal", "Bane",
#     "Benevolence", "Bias", "Bigotry", "Blasphemy", "Brevity",
#     "Calamity", "Candor", "Catharsis", "Censure", "Chaos", "Charisma",
#     "Chivalry", "Clemency", "Coercion", "Collusion", "Complacency",
#     "Concord", "Consensus", "Contempt", "Conundrum", "Credulity",
#     "Dearth", "Debacle", "Decorum", "Decree", "Deference", "Delusion",
#     "Demise", "Depravity", "Derision", "Despair", "Destiny",
#     "Detriment", "Devotion", "Dilemma", "Discord", "Disdain",
#     "Dissent", "Dogma", "Drudgery", "Duplicity", "Ebullience",
#     "Ecstasy", "Edict", "Efficacy", "Ego", "Elation", "Elegance",
#     "Empathy", "Enigma", "Enmity", "Ennui", "Epiphany", "Equity",
#     "Essence", "Euphoria", "Exodus", "Expediency", "Fallacy", "Fame",
#     "Famine", "Fatigue", "Feud", "Fidelity", "Finesse", "Flattery",
#     "Folly", "Fortitude", "Frenzy", "Friction", "Frugality",
#     "Futility", "Gallantry", "Gambit", "Genesis", "Glamour",
#     "Gluttony", "Gratitude"
# ]

words_list = [WORD_LIST_1, WORD_LIST_2]

NUM_STEPS = 10
NUM_INSTANCES = 100

# WORD_LIST = ["Aberration", "Acrimony", "Adulation"]

# NUM_STEPS = 3
# NUM_INSTANCES = 3


# ==========================================
# 3. CORE LOGIC (PARALLELIZED)
# ==========================================

def _worker_initial_gen(word):
    """Worker function to generate ONE initial description."""
    try:
        response = global_model.invoke([
            HumanMessage(f"Describe the object '{word}' in exactly 2 sentences without naming it directly.")
        ])
        return response.content.strip()
    except Exception as e:
        return "" # Return empty on fail, will be filtered later

def generate_initial_descriptions_parallel(word, count):
    """Generates initial descriptions using ThreadPool."""
    print(f"  > Generating {count} initial descriptions in parallel...")
    descriptions = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit 'count' tasks
        futures = [executor.submit(_worker_initial_gen, word) for _ in range(count)]

        for future in as_completed(futures):
            res = future.result()
            if res:
                descriptions.append(res)
            else:
                descriptions.append("ERROR")

    return descriptions

def _worker_process_step(desc):
    """
    Worker function to process ONE instance (Guess + Paraphrase).
    """
    if not desc or desc == "ERROR":
        return "ERROR", "ERROR"

    try:
        # 1. Probe (Guess)
        guess_resp = global_model.invoke([
            HumanMessage(f"Read this description: \"{desc}\"\nGuess the single noun being described. Reply with ONLY the word, no punctuation.")
        ])
        guess = guess_resp.content.strip().strip(".\"").lower()

        # 2. Chain (Paraphrase)
        paraphrase_resp = global_model.invoke([
            HumanMessage(f"Paraphrase the description: \"{desc}\"\nDo not explain.\nDo not guess the word.\nDo not add reasoning.\nOutput only the paraphrase.")
        ])
        new_desc = paraphrase_resp.content.strip()

        return guess, new_desc
    except Exception:
        return "ERROR", desc # Return ERROR for guess, but keep old desc to avoid breaking chain completely

def process_step_parallel(current_descriptions):
    """Runs the step for all instances in parallel."""
    guesses = [None] * len(current_descriptions)
    next_descriptions = [None] * len(current_descriptions)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures to their index so we keep order (Instance 1 stays Instance 1)
        future_to_index = {
            executor.submit(_worker_process_step, desc): i
            for i, desc in enumerate(current_descriptions)
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                g, d = future.result()
                guesses[i] = g
                next_descriptions[i] = d
            except Exception:
                guesses[i] = "ERROR"
                next_descriptions[i] = "ERROR"

    return guesses, next_descriptions

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":

    for list_idx, WORD_LIST in enumerate(words_list, 1):
        all_data_records = []

        print(f"\nStarting Parallel Experiment")
        print(f"Words: {len(WORD_LIST)} | Instances: {NUM_INSTANCES} | Steps: {NUM_STEPS}")
        print(f"Concurrency: {MAX_WORKERS} threads")

        # Use TQDM to track progress across words
        for word in tqdm(WORD_LIST, desc="Processing Words", unit="word"):

            # --- Step 0: Parallel Generation ---
            try:
                current_descs = generate_initial_descriptions_parallel(word, NUM_INSTANCES)
            except Exception as e:
                print(f"Skipping word '{word}' due to critical error: {e}")
                continue

            # Save Step 0
            for i, desc in enumerate(current_descs):
                all_data_records.append({
                    "Word": word,
                    "Instance_ID": i + 1,
                    "Step": 0,
                    "Description": desc,
                    "Guess": word
                })

            # --- Steps 1 to 10: Parallel Loop ---
            for step_num in range(1, NUM_STEPS + 1):
                # Run parallel processing for this step
                guesses, next_descs = process_step_parallel(current_descs)

                # Save data
                for i in range(NUM_INSTANCES):
                    all_data_records.append({
                        "Word": word,
                        "Instance_ID": i + 1,
                        "Step": step_num,
                        "Description": next_descs[i],
                        "Guess": guesses[i]
                    })

                # Update for next iteration
                current_descs = next_descs

            # Intermediate Save (Optional: saves after every word in case of crash)
            # remove this if you only want one file at the end
            if len(all_data_records) % 5000 == 0:
                    pd.DataFrame(all_data_records).to_csv("semantic_drift_below_hundred_words.csv", index=False)

        # --- Final Save ---
        if all_data_records:
            # 1. Create the "outputs" directory if it doesn't exist
            output_dir = "outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("\nExperiment Complete. Saving Final CSV...")
            df = pd.DataFrame(all_data_records)
            
            # 2. Extract model name for the file (clean 'google/' out if desired)
            # This uses the MODEL_NAME variable defined at the top
            clean_model_name = MODEL_NAME.split('/')[-1]
            
            # 3. Use enumerate index to add 1, 2, 3, 4 to the end
            # We get the index from the outer loop: for list_idx, WORD_LIST in enumerate(words_list, 1):
            output_filename = f"semantic_drift_experiment__{clean_model_name}-{list_idx}.csv"
            
            # 4. Save to the "outputs" folder
            save_path = os.path.join(output_dir, output_filename)
            
            df.to_csv(save_path, index=False)
            print(f"Success! Data saved to: {os.path.abspath(save_path)}")
            print(f"Total Rows: {len(df)}")
        else:
            print("No data was generated.")
