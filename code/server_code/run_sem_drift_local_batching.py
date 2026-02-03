import os
import re
import pandas as pd
from tqdm import tqdm
from models_local_gemma import LocalChatModel

class HumanMessage:
    def __init__(self, content: str):
        self.content = content

# Environment Configs
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(HERE, "word_categories.csv")
LOCAL_MODELS = {
    #"llama-3.2-3b-instruct": "/mnt/h200_raid5/zulfiyausmonova/models/llama-3.2-3b-instruct",
    "gemma-3-4b-it": "/mnt/h200_raid5/zulfiyausmonova/models/gemma-3-4b-it",
}
MODELS_TO_RUN = list(LOCAL_MODELS.keys())
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
NUM_INSTANCES = int(os.environ.get("NUM_INSTANCES", "100"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "140"))
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift")

global_model = None

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "", s.strip().replace(" ", "_"))

def get_clean_word_list(df, column_name):
    return [w.strip() for w in df[column_name].dropna().astype(str).tolist() if w.strip()]

def build_model(model_key: str) -> LocalChatModel:
    return LocalChatModel(
        model_path=LOCAL_MODELS[model_key],
        temperature=GEN_TEMPERATURE,
        max_new_tokens=GEN_MAX_NEW_TOKENS
    )

def generate_initial_descriptions(word: str, count: int):
    prompt = f"Describe the object '{word}' in exactly 2 sentences without naming it directly."
    # Batch generate all initial descriptions at once
    responses = global_model.invoke_batch([prompt] * count)
    return [r.content for r in responses]

def process_step_batched(current_descriptions):
    # 1. Batch Guessing
    guess_prompts = [
        f"Read this description: \"{d}\"\nGuess the single noun being described. Reply with ONLY the word, no punctuation."
        for d in current_descriptions
    ]
    guess_resps = global_model.invoke_batch(guess_prompts)
    guesses = [r.content.strip().split("\n")[0].strip(".\"").lower() for r in guess_resps]

    # 2. Batch Regeneration
    regen_prompts = []
    for d in current_descriptions:
        regen_prompts.append(
            f"Paraphrase the description: \"{d}\"\nDo not explain.\nDo not guess the word.\nDo not add reasoning.\nOutput only the paraphrase."
        )
    
    regen_resps = global_model.invoke_batch(regen_prompts)
    next_descriptions = [r.content for r in regen_resps]

    return guesses, next_descriptions

def run_category_experiment(model_key: str, category_name: str, word_list, output_dir: str):
    category_records = []
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{sanitize(model_key)}_{sanitize(category_name)}.csv")

    for word in tqdm(word_list, desc=f"Words in {category_name}"):
        # Step 0: Batch Generate
        current_descs = generate_initial_descriptions(word, NUM_INSTANCES)
        for i, desc in enumerate(current_descs):
            category_records.append({"Model": model_key, "Category": category_name, "Word": word, "Instance_ID": i + 1, "Step": 0, "Description": desc, "Guess": word})

        # Steps 1..NUM_STEPS: Batch Process
        for step_num in range(1, NUM_STEPS + 1):
            guesses, next_descs = process_step_batched(current_descs)
            for i in range(NUM_INSTANCES):
                category_records.append({"Model": model_key, "Category": category_name, "Word": word, "Instance_ID": i + 1, "Step": step_num, "Description": next_descs[i], "Guess": guesses[i]})
            current_descs = next_descs

        # Save checkpoint after every word
        pd.DataFrame(category_records).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(category_records)

if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    categories = df.columns.tolist()

    only_cats_env = os.environ.get("ONLY_CATEGORIES")
    if only_cats_env:
        only_set = {c.strip() for c in only_cats_env.split(",") if c.strip()}
        categories = [c for c in categories if c in only_set]

    for model_key in MODELS_TO_RUN:
        global_model = build_model(model_key)
        for category in categories:
            words = get_clean_word_list(df, category)
            result_df = run_category_experiment(model_key, category, words, OUTPUT_BASE)
            save_path = os.path.join(OUTPUT_BASE, f"Results_{sanitize(model_key)}_{sanitize(category)}.csv")
            result_df.to_csv(save_path, index=False)