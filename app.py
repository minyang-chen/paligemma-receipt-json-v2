import gradio as gr
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import spaces
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id="mychen76/paligemma-receipt-json-3b-mix-448-v2b"
dtype = torch.bfloat16
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

MAX_TOKENS = 512

import re
# let's turn that into JSON source from Donut
def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()
        output = {}
        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def modify_caption(caption: str) -> str:
    """
    Removes specific prefixes from captions.

    Args:
        caption (str): A string containing a caption.

    Returns:
        str: The caption with the prefix removed if it was present.
    """
    # Define the prefixes to remove
    prefix_substrings = [
        ('EXTRACT_JSON_RECEIPT', '')
    ]
    
    # Create a regex pattern to match any of the prefixes
    pattern = '|'.join([re.escape(opening) for opening, _ in prefix_substrings])
    replacers = {opening: replacer for opening, replacer in prefix_substrings}
    
    # Function to replace matched prefix with its corresponding replacement
    def replace_fn(match):
        return replacers[match.group(0)]
    
    # Apply the regex to the caption
    return re.sub(pattern, replace_fn, caption, count=1, flags=re.IGNORECASE)

def json_inference(image, input_text="EXTRACT_JSON_RECEIPT", device="cuda:0", max_new_tokens=512):
    inputs = processor(text=input_text, images=image, return_tensors="pt").to(device)
    # Autoregressively generate use greedy decoding here,for more fancy methods see https://huggingface.co/blog/how-to-generate
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Next turn each predicted token ID back into a string using the decode method
    # We chop of the prompt, which consists of image tokens and our text prompt
    image_token_index = model.config.image_token_index
    num_image_tokens = len(generated_ids[generated_ids==image_token_index])
    num_text_tokens = len(processor.tokenizer.encode(input_text))
    num_prompt_tokens = num_image_tokens + num_text_tokens + 2
    generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # convert it into JSON using the method below (taken from Donut):
    generated_json = token2json(generated_text)
    return generated_text, generated_json

# enable space
# @spaces.GPU
def create_captions_rich(image):   
    torch.cuda.empty_cache()
    prompt = "EXTRACT_JSON_RECEIPT"
    generated_text, generated_json = json_inference(image=image,input_text="EXTRACT_JSON_RECEIPT", device=device, max_new_tokens=MAX_TOKENS)
    return generated_json

css = """
  #mkd {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
  gr.HTML("<h1><center>PaliGemma Receipt and Invoice Model<center><h1>")
  with gr.Tab(label="Receipt or Invoices Image"):
    with gr.Row():
      with gr.Column():
        input_img = gr.Image(label="Input Picture")
        submit_btn = gr.Button(value="Submit")
      output = gr.Text(label="Receipt Json")

    gr.Examples([["receipt_image1.jpg"], ["receipt_image2.jpg"], ["receipt_image3.png"],["receipt_image4.png"]],
    inputs = [input_img],
    outputs = [output],
    fn=create_captions_rich,
    label='Try captioning on examples'
    )

    submit_btn.click(create_captions_rich, [input_img], [output])

demo.queue().launch(share=True,server_name="0.0.0.0",debug=True)