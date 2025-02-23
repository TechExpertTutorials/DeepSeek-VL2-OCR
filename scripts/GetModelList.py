from huggingface_hub import list_models

def get_models_by_multi_filters():
    models = list(list_models(
        task="image-classification",
        library="pytorch",
        trained_dataset="imagenet",
    ))
    if models:
        for model in models[0:10]:
            print(f"- {model.modelId}")

def get_hf_models(search_string):
    # Search for Qwen models
    hf_models = list(list_models(filter=search_string)) # list() to convert generator to list
    if hf_models:
        print(f"Found {len(hf_models)} models (may be paginated):")
        # Print some information about the the first 10 models
        for model in hf_models[:]: # Print first 10
            print(f"- {model.modelId}")  # Access attributes of the model objects

def get_gpt2_models():
    # Filter by model task category
    gpt_models = list(list_models(task="gpt2"))
    if gpt_models:
        print(f"Found {len(gpt_models)} GPT models. Printing First 10:")
        for model in gpt_models[0:10]:
            print(f"- {model.modelId}")
    else:
        print("No GPT models found.")

def get_top_10_models():
    top_10_models = list(list_models(sort="downloads", direction=-1, limit=10))
    if top_10_models:
        for model in top_10_models[1:10]:
            print(f"- {model.modelId}   {model.created_at}   {model.downloads}   {model.pipeline_tag}")

def main():
    get_qwen_models()

if __name__ == "__main__":
    main()