"""
conda create -n deepseek-ocr -c conda-forge -y  
conda activate deepseek-ocr   
conda install python pillow ipykernel jupyter nb_conda_kernels ipywidgets -c conda-forge -y  
pip install git+https://github.com/huggingface/transformers  

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  

# pip3 install torch=2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
 


# conda install pytorch=2.0.1 -c conda-forge
# or try pip3 install torch=2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install xformers==0.0.21
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# to avoid this error from the DeepSeek-VL2 repo: 
# Could not find a version that satisfies the requirement torch==2.0.1 (from deepseek-vl2)

pip install deepseek_vl2
pip install git+https://github.com/deepseek-ai/DeepSeek-VL2

git clone https://github.com/deepseek-ai/DeepSeek-VL2
cd DeepSeek-VL2
pip install -e .
"""

import torch

if torch.cuda.is_available():
    print("CUDA is available! 🎉")
    device = torch.device("cuda")  # Set device to CUDA
else:
    print("CUDA is NOT available. 🙁")
    device = torch.device("cpu")  # Set device to CPU (or raise an exception)

# automatic mixed precision - lowers precision (FP16 or BF16) to use less memory
import torch.cuda.amp as amp
scaler = amp.GradScaler() # Initialize a scaler

from transformers import AutoModelForCausalLM
from PIL import Image
# think we should be using AutoModelForVision2Seq instead!!

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
# model_path = "deepseek-ai/deepseek-vl2"
# model_path = "deepseek-ai/deepseek-vl2-small"
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# downloads the model to cache folder: C:\Users\techexpert\.cache\huggingface\hub
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
# or change order, device should be set first!
vl_gpt = vl_gpt.to(device) # Move to GPU first
vl_gpt = vl_gpt.to(torch.bfloat16)  # Then change precision
vl_gpt.eval() # Set to eval mode

image = Image.open('animals.jpg').convert('RGB')
## single image conversation example
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n|ref|>Describe the image.<|/ref|>.",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]


## multiple images (or in-context learning) conversation example
# conversation = [
#     {
#         "role": "User",
#         "content": "A dog wearing nothing in the foreground, "
#                    "a dog wearing a santa hat, "
#                    "a dog wearing a wizard outfit, and "
#                    "what's the dog wearing?",
#         "images": [
#             "images/dog_a.png",
#             "images/dog_b.png",
#             "images/dog_c.png",
#             "images/dog_d.png",
#         ],
#     },
#     {"role": "Assistant", "content": ""}
# ]

# load images and prepare for inputs
# also move the model to GPU
model_inputs = vl_chat_processor(
    conversations=conversation,
    images=[image],
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**model_inputs)

"""
"small" model
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 42.00 MiB (GPU 0; 8.00 GiB total capacity; 32.75 GiB already allocated; 0 bytes free; 
38.40 GiB reserved in total by PyTorch) 
If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
"""

# run the model to get the response
# Important for inference: 
# Don't calculate gradients as the model parameters are not updated during inference, 
# and this will save time and memory 
with torch.no_grad():
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=model_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
        do_sample=False,
        use_cache=True
    )

# generated_text = vl_chat_processor.decode(outputs[0], skip_special_tokens=True)  # No .cpu()!
# print(generated_text)

# response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
response = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
print(response)
# print(f"{model_inputs['sft_format'][0]}", answer)

print("finished")