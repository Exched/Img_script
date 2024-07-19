import spaces
import gradio as gr
import os
import random
import json
import time
import uuid
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image, DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import torch
from typing import Tuple
from datetime import datetime
import requests
import torch
from diffusers import DiffusionPipeline
import importlib
from urllib.parse import urlparse

random.seed(time.time())
MAX_SEED = 12211231
CACHE_EXAMPLES = "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4192"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

NUM_IMAGES_PER_PROMPT = 1


cfg = json.load(open("app.conf"))

def load_pipeline_and_scheduler():
    clip_skip = cfg.get("clip_skip", 0)

    # Download the model files
    ckpt_dir = snapshot_download(repo_id=cfg["model_id"])

    # Load the models
    vae = AutoencoderKL.from_pretrained(os.path.join(ckpt_dir, "vae"), torch_dtype=torch.float16)
   
    pipe = StableDiffusionXLPipeline.from_pretrained(
        ckpt_dir,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    pipe.unet.set_attn_processor(AttnProcessor2_0())

    # Define samplers
    samplers = {
        "Euler a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        "DPM++ SDE Karras": DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    }
    # Set the scheduler based on the selected sampler
    pipe.scheduler = samplers[cfg.get("sampler","DPM++ SDE Karras")]
    
    # Set clip skip
    pipe.text_encoder.config.num_hidden_layers -= (clip_skip - 1)

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")
    return pipe
pipe = load_pipeline_and_scheduler()
css = '''
.gradio-container{max-width: 560px !important}
body {
    background-color: rgb(3, 7, 18);
    color: white;
}
.gradio-container {
    background-color: rgb(3, 7, 18) !important;
    border: none !important;
}
footer {display: none !important;}
''' 
js = '''
<script src="https://huggingface.co/spaces/nsfwalex/sd_card/resolve/main/prompt.js"></script>
<script>
function getEnvInfo() {
    const result = {};
    // Get URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    for (const [key, value] of urlParams) {
        result[key] = value;
    }

    // Get current domain and convert to lowercase
    result["__domain"] = window.location.hostname.toLowerCase();

    // Get iframe parent domain, if any, and convert to lowercase
    try {
        if (window.self !== window.top) {
            result["__iframe_domain"] = document.referrer 
                ? new URL(document.referrer).hostname.toLowerCase()
                : "unable to get iframe parent domain";
        }else{
            result["__iframe_domain"] = "";
        }
    } catch (e) {
        result["__iframe_domain"] = "unable to access iframe parent domain";
    }

    return result;
}
function isValidEnv(){
    envInfo = getEnvInfo();
    return envInfo["e"] == "1" || 
        envInfo["__domain"].indexOf("nsfwais.io") != -1 || 
        envInfo["__iframe_domain"].indexOf("nsfwais.io") != -1 ||
        envInfo["__domain"].indexOf("127.0.0.1") != -1 || 
        envInfo["__iframe_domain"].indexOf("127.0.0.1") != -1;
}
window.g=function(p){ 
  params = getEnvInfo();
  if (!isValidEnv()){
      return "";
  }
  const conditions = {
    "tag": ["normal", "sexy", "porn"],
    "exclude_category": ["Clothing"],
    "count_per_tag": 1
  };
  prompt = generateSexyPrompt()
  console.log(prompt);
  return prompt
}

window.postMessageToParent = function(prompt, event, source, value) {
    // Construct the message object with the provided parameters
    console.log("post start",event, source, value);
    const message = {
        event: event,
        source: source,
        value: value
    };
    if (!prompt){
        prompt = window.g();
        
        // Find the prompt input element
        const promptContainer = document.getElementById('prompt_input_box');
        if (promptContainer) {
            const promptInput = promptContainer.querySelector('input') || promptContainer.querySelector('textarea');
            if (promptInput) {
                promptInput.value = prompt;
                // Trigger an input event to ensure Gradio recognizes the change
                promptInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    }
    if (window.self !== window.top) {
        // Post the message to the parent window
        window.parent.postMessage(message, '*');
    }else if(isValidEnv()){
        try{
            sendCustomEventToDataLayer({},event,source,value)
        } catch (error) {
            console.error("Error in sendCustomEventToDataLayer:", error);
        }
    }else{
        console.log("Not in an iframe, can't post to parent");
    }
    console.log("post finish");
    return prompt;
}
function uploadImage(prompt, images, event, source, value) {
    // Ensure we're in an iframe
    console.log("uploadImage", prompt, images && images.length > 0 ? images[0].image.url : null, event, source, value);
    // Get the first image from the gallery (assuming it's an array)
    let imageUrl = images && images.length > 0 ? images[0].image.url : null;

    if (window.self !== window.top) {
        // Post the message to the parent window
        // Prepare the data to send
        let data = {
            event: event,
            source: source,
            value:{
                prompt: prompt,
                image: imageUrl
            }
        };
        window.parent.postMessage(JSON.stringify(data), '*');
    } else if (isValidEnv()){
        try{
            sendCustomEventToDataLayer({},event,source,{"prompt": prompt, "image":imageUrl})
        } catch (error) {
            console.error("Error in sendCustomEventToDataLayer:", error);
        }
    }else{
        console.log("Not in an iframe, can't post to parent");
    }
    return ""
}
function onDemoLoad(){
    let envInfo = getEnvInfo();
    console.log(envInfo);
    if (isValidEnv()){
        var element = document.getElementById("desc_html_code");
        if (element) {
            element.parentNode.removeChild(element);
        }
    }
    return;
    //return envInfo["__domain"], envInfo["__iframe_domain"]
}
</script>
'''
desc_html='''
<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center; margin-top: 20px;">
  <p style="font-size: 16px; color: #333;">
    For the full version and more exciting NSFW AI apps, visit 
    <a href="https://nsfwais.io?utm_source=hf_'''+cfg["model_id"].replace("/","_")+'''&utm_medium=referral" style="color: #0066cc; text-decoration: none; font-weight: bold;" rel="dofollow">nsfwais.io</a>!
  </p>
</div>
'''
def save_image(img):
    # Generate a unique filename
    unique_name = str(uuid.uuid4()) + ".webp"
    
    # Convert the image to WebP format
    webp_img = img.convert("RGB")  # Ensure the image is in RGB mode
    
    # Save the image in WebP format with high quality
    webp_img.save(unique_name, "WEBP", quality=90)
    
    # Open the saved WebP file and return it as a PIL Image object
    with Image.open(unique_name) as webp_file:
        webp_image = webp_file.copy()
    
    return webp_image, unique_name
    
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=60)
def generate(p, progress=gr.Progress(track_tqdm=True)):
    negative_prompt = cfg.get("negative_prompt", "")
    style_selection = ""
    use_negative_prompt = True
    seed = 0
    width = cfg.get("width", 1024)
    height = cfg.get("width", 768) 
    inference_steps = cfg.get("inference_steps", 30)
    randomize_seed = True
    guidance_scale = cfg.get("guidance_scale", 7.5)
    prompt_str = cfg.get("prompt", "{prompt}").replace("{prompt}", p)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(pipe.device).manual_seed(seed)
        
    images = pipe(
        prompt=prompt_str,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        output_type="pil",
    ).images
    images = [save_image(img) for img in images]
    image_paths = [i[1] for i in images]
    print(prompt_str, image_paths)
    return [i[0] for i in images]

with gr.Blocks(css=css,head=js,fill_height=True) as demo:
    with gr.Row(equal_height=False):
        with gr.Group():
            gr.HTML(value=desc_html, elem_id='desc_html_code')
            result = gr.Gallery(
              label="Result",  show_label=False, columns=1, rows=1, show_share_button=True,
              show_download_button=True,allow_preview=True,interactive=False, min_width=cfg.get("window_min_width", 340),height=360
            )
            with gr.Row(): 
                prompt = gr.Text(
                    show_label=False,
                    max_lines=2,
                    lines=2,
                    placeholder="Enter what you want to see",
                    container=False,
                    scale=5,
                    min_width=100,
                    elem_id="prompt_input_box"
                )
                random_button = gr.Button("Surprise Me", scale=1, min_width=10)
            run_button = gr.Button( "GO!", scale=1, min_width=20, variant="primary",icon="https://huggingface.co/spaces/nsfwalex/sd_card/resolve/main/hot.svg")
        
    def on_demo_load(request: gr.Request):
        current_domain = request.request.headers.get("Host", "")
    
        # Get the potential iframe parent domain from the Referer header
        referer = request.request.headers.get("Referer", "")
        iframe_parent_domain = ""
    
        if referer:
            try:
                parsed_referer = urlparse(referer)
                iframe_parent_domain = parsed_referer.netloc
            except:
                iframe_parent_domain = "Unable to parse referer"

        params = dict(request.query_params)
        default_image = cfg.get("cover_path", None)
        
        if default_image:
            if isinstance(default_image, list):
                # Filter out non-existent paths
                existing_images = [img for img in default_image if os.path.exists(img)]
                #print(f"found cover files: {existing_images}")
                if existing_images:
                    default_image = random.choice(existing_images)
                else:
                    default_image = None
            elif not os.path.exists(default_image):
                print(f"cover file not existed, {default_image}")
                default_image = None
        else:
            default_image = None
        print(f"load_demo, urlparams={params},cover={default_image},domain={current_domain},iframe={iframe_parent_domain}")
        if params.get("e", "0") == "1" or "nsfwais.io" in current_domain or "nsfwais.io" in iframe_parent_domain or "127.0.0.1" in current_domain or "127.0.0.1" in iframe_parent_domain:
            #update the image
            #bind events
            return [Image.open(default_image)]
        return []
            

    result.change(fn=lambda :None , inputs=[prompt,result], outputs=[], js=f'''(p,img)=>window.uploadImage(p, img,"process_finished","demo_hf_{cfg.get("name")}_card", "{cfg["model_id"]}")''')    
    run_button.click(generate, inputs=[prompt], outputs=[result],trigger_mode="once",js=f'''(p)=>window.postMessageToParent(p,"process_started","demo_hf_{cfg.get("name")}_card", "click_go")''')
    random_button.click(fn=lambda x:x, inputs=[prompt], outputs=[prompt], js='''(p)=>window.g(p)''')
    demo.load(fn=on_demo_load, inputs=[], outputs=[result], js='''()=>onDemoLoad()''')
if __name__ == "__main__":
    demo.queue(max_size=100).launch(show_api=False)