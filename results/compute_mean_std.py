import yaml
import numpy as np
import pandas as pd


yaml_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/results/results.yaml"

with open(yaml_path, "r") as f:
    results = yaml.safe_load(f)

def extract_stats(results):
    records = []
    for model_name, seeds in results.items():
 
        for seed, metrics in seeds.items():
            for task, value in metrics.items():
                if isinstance(value, (int, float)): 
                    records.append({
                        "model": model_name,
                        "task": task,
                        "seed": seed,
                        "value": value
                    })

            if "multitask" in metrics:
                for subtask, value in metrics["multitask"].items():
                    records.append({
                        "model": model_name,
                        "task": f"multitask_{subtask}",
                        "seed": seed,
                        "value": value
                    })
    return pd.DataFrame(records)

df = extract_stats(results)

summary = df.groupby(["model", "task"])["value"].agg(["mean", "std"]).reset_index()

print("\n=== Results ===")
print(summary.to_string(index=False))


summary.to_csv("/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/results/results_summary.csv", index=False)