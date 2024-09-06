# paligemma-receipt-json-v2
demo usage of paligemma extraction of receipt image to json object

### Model and Task
```
FINETUNED_MODEL_ID="mychen76/paligemma-receipt-json-3b-mix-448-v2b"
TASK_PROMPT = "EXTRACT_JSON_RECEIPT"
```
### Usage
setup
```
pip -r install requirements.txt
```
run gradio app
```
gradio app.py
```
next launch the browser and enter url display in console.
