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

# Your Nebius API Key
os.environ["NEBIUS_API_KEY"] = "v1.CmMKHHN0YXRpY2tleS1lMDByNG13OHJlOTEwYXhtZjcSIXNlcnZpY2VhY2NvdW50LWUwMHZ2ZW53eDUwMDM1NTU1NjIMCOeCosgGEPu9g74COgsI54W6kwcQwMLuSkACWgNlMDA.AAAAAAAAAAF-s3IVuPd-6SwZfzos0vgqlAlUZtfge6Kj5JAVVepABWajqetR76LusvMMN1mo0E5Y5TbLdhzBkjxNaiMXxrQM"
# Files
INPUT_FILE = "word_categories.csv"  # Ensure this matches your file name

# Models to iterate through
# Note: Verified correct Nebius ID for Gemma is usually 'google/gemma-2-9b-it'
MODELS = [
    #"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-fast"
]

# Experiment Settings
MAX_WORKERS = 30
NUM_STEPS = 10
NUM_INSTANCES = 100

# Global placeholder (will be updated in the loop)
global_model = None

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def build_model(model_name):
    """Creates the ChatOpenAI client for the specific model."""
    print(f"\n--- Initializing Model: {model_name} ---")
    return ChatOpenAI(
        base_url="https://api.studio.nebius.ai/v1",
        api_key=os.environ["NEBIUS_API_KEY"],
        model=model_name,
        temperature=0.7,
        max_retries=3,
        request_timeout=30
    )

def get_clean_word_list(df, column_name):
    """Extracts a clean list of words from a specific dataframe column."""
    # Drop NaNs and convert to string
    words = df[column_name].dropna().astype(str).tolist()
    # clean whitespace
    return [w.strip() for w in words if w.strip()]

# ==========================================
# 3. WORKER FUNCTIONS
# ==========================================

def _worker_initial_gen(word):
    """Worker function to generate ONE initial description."""
    try:
        response = global_model.invoke([
            HumanMessage(f"Describe the object '{word}' in exactly 2 sentences without naming it directly.")
        ])
        return response.content.strip()
    except Exception:
        return ""

def generate_initial_descriptions_parallel(word, count):
    """Generates initial descriptions using ThreadPool."""
    descriptions = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_worker_initial_gen, word) for _ in range(count)]
        for future in as_completed(futures):
            res = future.result()
            descriptions.append(res if res else "ERROR")
    return descriptions

def _worker_process_step(desc):
    """
    Worker function to process ONE instance.
    Logic: Description -> Guess -> HYBRID NEW Description
    """
    if not desc or desc == "ERROR":
        return "ERROR", "ERROR"

    try:
        # 1. PROBE
        guess_resp = global_model.invoke([
            HumanMessage(f"Read this description: \"{desc}\"\nGuess the single noun being described. Reply with ONLY the word, no punctuation.")
        ])
        guess = guess_resp.content.strip().strip(".\"").lower()
        clean_guess = guess.split("\n")[0].strip()

        # 2. HYBRID REGENERATION
        if clean_guess and len(clean_guess) < 50:
            new_desc_resp = global_model.invoke([
                HumanMessage(
                    f"The word '{clean_guess}' was guessed from the following description: \"{desc}\"\n\n"
                    f"Write a NEW, exactly 2-sentence description of '{clean_guess}' that incorporates the specific details or style "
                    f"of that previous description. Do not name the object directly."
                )
            ])
            new_desc = new_desc_resp.content.strip()
        else:
            new_desc = "ERROR"

        return clean_guess, new_desc

    except Exception:
        return "ERROR", "ERROR"

def process_step_parallel(current_descriptions):
    """Runs the step for all instances in parallel."""
    guesses = [None] * len(current_descriptions)
    next_descriptions = [None] * len(current_descriptions)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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

def run_category_experiment(model_name, category_name, word_list):
    """Runs the full experiment for one category."""
    category_records = []
    print(f"\n>>> Starting Category: {category_name} ({len(word_list)} words)")

    for word in tqdm(word_list, desc=f"Words in {category_name}", unit="word"):
        # Step 0
        try:
            current_descs = generate_initial_descriptions_parallel(word, NUM_INSTANCES)
        except Exception as e:
            print(f"Error on word {word}: {e}")
            continue

        for i, desc in enumerate(current_descs):
            category_records.append({
                "Model": model_name,
                "Category": category_name,
                "Word": word,
                "Instance_ID": i + 1,
                "Step": 0,
                "Description": desc,
                "Guess": word
            })

        # Steps 1 to 10
        for step_num in range(1, NUM_STEPS + 1):
            guesses, next_descs = process_step_parallel(current_descs)

            for i in range(NUM_INSTANCES):
                category_records.append({
                    "Model": model_name,
                    "Category": category_name,
                    "Word": word,
                    "Instance_ID": i + 1,
                    "Step": step_num,
                    "Description": next_descs[i],
                    "Guess": guesses[i]
                })

            current_descs = next_descs

    return pd.DataFrame(category_records)



#==============================================
#  4. Main function
#==============================================

if __name__ == "__main__":

    # 1. SETUP OUTPUT DIRECTORY
    # This creates the "outputs" folder in your current directory if it doesn't exist
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        categories = df.columns.tolist()
        print(f"Found categories: {categories}")
    except Exception as e:
        print(f"Failed to read file: {e}")
        exit()

    # Iterate Models
    for model_name in MODELS:
        # Update Global Model for Workers
        global_model = build_model(model_name)

        # Sanity Check
        try:
            test = global_model.invoke([HumanMessage("Test")])
            print("API Connection OK.")
        except Exception as e:
            print(f"API Error for {model_name}: {e}")
            continue

        # Iterate Categories
        for category in categories:
            words = get_clean_word_list(df, category)

            # Run Experiment
            result_df = run_category_experiment(model_name, category, words)

            # Save File Locally immediately
            sanitized_model = model_name.split("/")[-1].replace("-", "_")
            sanitized_cat = category.replace(" ", "_").replace("-", "_")
            filename = f"Results_{sanitized_model}_{sanitized_cat}.csv"

            save_path = os.path.join(output_dir, filename)
            
            result_df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")

    print("\nAll experiments complete!")