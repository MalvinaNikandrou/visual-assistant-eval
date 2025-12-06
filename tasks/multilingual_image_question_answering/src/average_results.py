import json
from pathlib import Path

results_dir = Path("tasks/multilingual_image_question_answering/results")

model_name_map = {
    "allenai": "Molmo-7B-D-0924",
    "google": "Paligemma-3b-mix-448",
    "HuggingFaceM4": "Idefics2-8b",
    "llava-hf": "Llava-v1.6-7B-hf",
    "meta-llama": "Llama-3.2-11B-Vision-Instruct",
    "microsoft": "Phi-3-vision-128k-instruct",
    "openbmb": "MiniCPM-Llama-2_5",
    "OpenGVLab": "InternVL2.5-8B-MPO",
    "Qwen": "Qwen2-VL-7B-Instruct",
}
# Language, Language code, Script, Resource
lang_metadata = [
    ["Arabic", "ar", "Arabic", "High"],
    ["Bengali", "bn", "Bengali", "Mid"],
    ["Czech", "cs", "Latin", "High"],
    ["Danish", "da", "Latin", "Mid"],
    ["German", "de", "Latin", "High"],
    ["Greek", "el", "Greek", "Mid"],
    ["English", "en", "Latin", "High"],
    ["Spanish", "es", "Latin", "High"],
    ["Persian", "fa", "Arabic", "High"],
    ["Finnish", "fi", "Latin", "High"],
    ["Filipino", "fil", "Latin", "Mid"],
    ["French", "fr", "Latin", "High"],
    ["Hebrew", "he", "Hebrew", "Mid"],
    ["Hindi", "hi", "Devanagari", "High"],
    ["Croatian", "hr", "Latin", "Mid"],
    ["Hungarian", "hu", "Latin", "High"],
    ["Indonesian", "id", "Latin", "Mid"],
    ["Italian", "it", "Latin", "High"],
    ["Japanese", "ja", "Japanese", "High"],
    ["Korean", "ko", "Hangul", "High"],
    ["Māori", "mi", "Latin", "Low"],
    ["Dutch", "nl", "Latin", "High"],
    ["Norwegian", "no", "Latin", "Low"],
    ["Polish", "pl", "Latin", "High"],
    ["Portuguese", "pt", "Latin", "High"],
    ["Romanian", "ro", "Latin", "Mid"],
    ["Russian", "ru", "Cyrillic", "High"],
    ["Swedish", "sv", "Latin", "High"],
    ["Swahili", "sw", "Latin", "Low"],
    ["Telugu", "te", "Telugu", "Low"],
    ["Thai", "th", "Thai", "Mid"],
    ["Turkish", "tr", "Latin", "High"],
    ["Ukrainian", "uk", "Cyrillic", "Mid"],
    ["Vietnamese", "vi", "Latin", "High"],
    ["Chinese", "zh", "Han", "High"],
]
code_to_metadata = {}
for metadata in lang_metadata:
    code_to_metadata[metadata[1]] = {
        "name": metadata[0],
        "code": metadata[1],
        "script": metadata[2],
        "resource": metadata[3],
    }

results_per_model = {}
results_per_model_per_resource = {"High": {}, "Mid": {}, "Low": {}}
results_per_model_per_script = {
    "Latin": {},
    "Han": {},
    "Japanese": {},
    "Hangul": {},
    "Cyrillic": {},
    "Arabic": {},
    "Devanagari": {},
    "Hebrew": {},
    "Thai": {},
    "Telugu": {},
    "Greek": {},
    "Bengali": {},
}
languages = [code_to_metadata[lang.name]["code"] for lang in results_dir.iterdir()]
languages.sort()
print("Read results per language")
for model in model_name_map:
    print(f"\n{model_name_map[model]}")
    results = []
    results_per_model[model] = []

    for lang in languages:
        lang_dir = results_dir.joinpath(lang)
        covered = False
        for filename in lang_dir.joinpath(model).iterdir():
            # if it is a csv continue
            if filename.suffix == ".json":
                covered = True
                with open(filename, "r") as f:
                    acc = json.load(f)["overall"]
                results.append(acc)
                results_per_model[model].append(acc)
                if model not in results_per_model_per_resource[code_to_metadata[lang]["resource"]]:
                    results_per_model_per_resource[code_to_metadata[lang]["resource"]][model] = []
                results_per_model_per_resource[code_to_metadata[lang]["resource"]][model].append(acc)
                if model not in results_per_model_per_script[code_to_metadata[lang]["script"]]:
                    results_per_model_per_script[code_to_metadata[lang]["script"]][model] = []
                results_per_model_per_script[code_to_metadata[lang]["script"]][model].append(acc)
        if not covered:
            print(f"Missing: {lang} - {model}")

# Print average results per model
print("Average results per model")
for model in model_name_map:
    print(f"{model_name_map[model]}: {sum(results_per_model[model]) / len(results_per_model[model]) * 100:.1f}")

# Print average results per resource
print("\nAverage results per resource")
for model in model_name_map:
    str = [f"{model_name_map[model]}"]
    for resource in ["High", "Mid", "Low"]:
        perf = f"{sum(results_per_model_per_resource[resource][model]) / len(results_per_model_per_resource[resource][model]) * 100:.1f}"
        str.append(perf)
    print(" & ".join(str), "\\\\")

# Print average results per script
print("\nAverage results per script")
scripts = list(results_per_model_per_script.keys())
print(" & ".join([""] + scripts), "\\\\")
for model in model_name_map:
    str = [f"{model_name_map[model]}"]
    for script in scripts:
        perf = f"{sum(results_per_model_per_script[script][model]) / len(results_per_model_per_script[script][model]) * 100:.1f}"
        str.append(perf)
    print(" & ".join(str), "\\\\")

# Print analytic results
print("\nAnalytic results")
print(" & ".join([""] + languages[:18]), "\\\\")
for model in model_name_map:
    str = [f"{model_name_map[model]}"]
    for idx, lang in enumerate(languages[:18]):
        str.append(f"{results_per_model[model][idx] * 100:.1f}")
    print(" & ".join(str))

print(" & ".join([""] + languages[18:]), "\\\\")
for model in model_name_map:
    str = [f"{model_name_map[model]}"]
    for idx, lang in enumerate(languages[18:], 18):
        str.append(f"{results_per_model[model][idx] * 100:.1f}")
    print(" & ".join(str), "\\\\")
