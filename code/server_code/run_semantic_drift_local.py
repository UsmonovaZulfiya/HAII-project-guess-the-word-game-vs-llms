import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# from langchain_core.messages import HumanMessage
from models_local import LocalChatModel

# ==========================================
# CONFIG
# ==========================================

class HumanMessage:
    def __init__(self, content: str):
        self.content = content

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(HERE, "word_categories.csv")

# Your current local model folders:
LOCAL_MODELS = {
    "gemma-2-9b-it": "/mnt/h200_raid5/zulfiyausmonova/models/gemma-2-9b-it",
    "llama-3.1-8b-instruct": "/mnt/h200_raid5/zulfiyausmonova/models/llama-3.1-8b-instruct",
}

# Run both in one job by default
MODELS_TO_RUN = list(LOCAL_MODELS.keys())

# Experiment settings
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
NUM_INSTANCES = int(os.environ.get("NUM_INSTANCES", "100"))

# Generation settings
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "140"))

# Output to big disk (recommended)
OUTPUT_BASE = os.environ.get(
    "OUTPUT_BASE",
    "/mnt/disk1/zulfiyausmonova/outputs/semantic_drift"
)

# Global placeholder for your existing function style
global_model = None


def sanitize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s


def get_clean_word_list(df, column_name):
    words = df[column_name].dropna().astype(str).tolist()
    return [w.strip() for w in words if w.strip()]


def build_model(model_key: str) -> LocalChatModel:
    model_path = LOCAL_MODELS[model_key]
    print(f"\n--- Loading local model: {model_key} ---")
    return LocalChatModel(
        model_path=model_path,
        temperature=GEN_TEMPERATURE,
        max_new_tokens=GEN_MAX_NEW_TOKENS
    )


# ==========================================
# WORKERS (SEQUENTIAL, GPU-safe)
# ==========================================

def _worker_initial_gen(word: str) -> str:
    try:
        response = global_model.invoke([
            HumanMessage(f"Describe the object '{word}' in exactly 2 sentences without naming it directly.")
        ])
        return response.content.strip()
    except Exception:
        return ""


def generate_initial_descriptions(word: str, count: int):
    descriptions = []
    for _ in range(count):
        res = _worker_initial_gen(word)
        descriptions.append(res if res else "ERROR")
    return descriptions


def _worker_process_step(desc: str):
    if not desc or desc == "ERROR":
        return "ERROR", "ERROR"

    try:
        # 1) Guess
        guess_resp = global_model.invoke([
            HumanMessage(
                f"Read this description: \"{desc}\"\n"
                f"Guess the single noun being described. Reply with ONLY the word, no punctuation."
            )
        ])
        guess = guess_resp.content.strip().strip(".\"").lower()
        clean_guess = guess.split("\n")[0].strip()

        # 2) Regenerate description
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


def process_step(current_descriptions):
    guesses, next_descriptions = [], []
    for desc in current_descriptions:
        g, d = _worker_process_step(desc)
        guesses.append(g)
        next_descriptions.append(d)
    return guesses, next_descriptions


def run_category_experiment(model_key: str, category_name: str, word_list, output_dir: str):
    category_records = []
    print(f"\n>>> Starting Category: {category_name} ({len(word_list)} words)")

    checkpoint_path = os.path.join(
        output_dir, f"checkpoint_{sanitize(model_key)}_{sanitize(category_name)}.csv"
    )

    for word in tqdm(word_list, desc=f"Words in {category_name}", unit="word"):
        # Step 0
        current_descs = generate_initial_descriptions(word, NUM_INSTANCES)

        for i, desc in enumerate(current_descs):
            category_records.append({
                "Model": model_key,
                "Category": category_name,
                "Word": word,
                "Instance_ID": i + 1,
                "Step": 0,
                "Description": desc,
                "Guess": word
            })

        # Steps 1..NUM_STEPS
        for step_num in range(1, NUM_STEPS + 1):
            guesses, next_descs = process_step(current_descs)

            for i in range(NUM_INSTANCES):
                category_records.append({
                    "Model": model_key,
                    "Category": category_name,
                    "Word": word,
                    "Instance_ID": i + 1,
                    "Step": step_num,
                    "Description": next_descs[i],
                    "Guess": guesses[i]
                })

            current_descs = next_descs

        # checkpoint after each word
        pd.DataFrame(category_records).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(category_records)


if __name__ == "__main__":

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print(f"Loading data from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    categories = df.columns.tolist()
    print(f"Found categories: {categories}")

    # Optionally restrict categories via env var (comma-separated)
    only_cats_env = os.environ.get("ONLY_CATEGORIES")  # e.g. "Animals,Furniture"
    if only_cats_env:
        only_set = {c.strip() for c in only_cats_env.split(",") if c.strip()}
        categories = [c for c in categories if c in only_set]
        print(f"Restricted to categories: {categories}")

    # Run BOTH models sequentially in the same job
    for model_key in MODELS_TO_RUN:
        global_model = build_model(model_key)

        # Sanity check
        try:
            test = global_model.invoke([HumanMessage("Reply with OK only.")]).content
            print(f"Sanity check output: {test[:80]}")
        except Exception as e:
            print(f"Sanity check failed for {model_key}: {e}")
            continue

        for category in categories:
            words = get_clean_word_list(df, category)
            result_df = run_category_experiment(model_key, category, words, OUTPUT_BASE)

            filename = f"Results_{sanitize(model_key)}_{sanitize(category)}.csv"
            save_path = os.path.join(OUTPUT_BASE, filename)
            result_df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")

    print("All experiments complete.")
