from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from utils import transform_ner_output, get_device


device = get_device()
model_path = "models/ner_model"
model_name = "Mu7annad/ner-xlm-roberta-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
classifier.save_pretrained(model_path)
text = "Hi my name is Muhannad Ashraf and i live in Cairo"

output = classifier(text)
print(transform_ner_output(output))
