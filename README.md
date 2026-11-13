## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Describing images by hand is slow, and many applications need image descriptions automatically, such as accessibility tools, photo organisation and content search. This experiment develops an interactive image-captioning application. It accepts an image from the user, processes it with the pretrained BLIP model, and generates a meaningful text description. The application uses a simple Gradio interface, so it is easy to use and to evaluate

### DESIGN STEPS:

#### STEP 1:Install and import the required libraries

Install the packages transformers, torch, Pillow and gradio.
Import the BLIP processor and model from the Transformers library.

#### STEP 2:Load the pretrained BLIP model

Load the pretrained BLIP image-captioning model (Salesforce/blip-image-captioning-base) and its processor.
Convert the uploaded image to the required format (RGB, resized) and pass it to the model. The model generates the caption, which the processor decodes into text.

#### STEP 3:Develop and launch the Gradio application

Create a Gradio interface with an image-upload component as the input and a text box as the output.
When the user uploads an image and clicks Submit, the BLIP model generates a caption, and the caption is displayed in the output box.s

### PROGRAM:
```python
import sys, subprocess, importlib.util

def ensure(pkg, module=None, extra=None):
    """Install a package only if it is missing."""
    if importlib.util.find_spec(module or pkg) is None:
        print(f"Installing {pkg} ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-warn-script-location", pkg] + (extra or []),
                           capture_output=True, text=True)
        print("  return code:", r.returncode)   # 0 means success
    else:
        print(f"{pkg} already installed")

ensure("torch", extra=["--index-url", "https://download.pytorch.org/whl/cpu"])
ensure("transformers")
ensure("pillow", "PIL")
print("Done")
```

```python
import site
site.addsitedir(site.getusersitepackages())   # lets the kernel see user-installed packages

import torch
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration

model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
model.eval()
print("BLIP model loaded")

def captioner(image):
    if image is None:
        return "Please upload an image."
    try:
        image = image.convert("RGB")
        image.thumbnail((512, 512))          # shrink big photos to save memory
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=40)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {e}"

gr.close_all()
demo = gr.Interface(
    fn=captioner,
    inputs=[gr.Image(label="Upload image", type="pil")],
    outputs=[gr.Textbox(label="Caption")],
    title="Image Captioning with BLIP",
    description="Caption any image using the BLIP model",
    allow_flagging="never",
)
demo.launch(share=True) 
```
### OUTPUT:
<img width="1890" height="971" alt="image" src="https://github.com/user-attachments/assets/ba47287c-4d5c-4a89-a00d-b9b12b515f74" />

### RESULT:
The image-captioning prototype was built using the pretrained BLIP model and a Gradio interface. For each uploaded image, the application generated a short, relevant caption, so the prototype was successfully developed and evaluated.
