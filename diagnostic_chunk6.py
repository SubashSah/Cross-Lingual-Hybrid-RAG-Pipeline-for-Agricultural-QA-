import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
nepali_token_id = tokenizer.convert_tokens_to_ids("npi_Deva")   

with open(
    "storage/chunks/original_chunks.pkl",
    "rb"
) as f:
    original_chunks = pickle.load(f)

text = original_chunks[6].page_content

text = text.replace("–", "-")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)

segments = splitter.split_text(text)


translated_segments = []

for i, segment in enumerate(segments):

    print(f"Translating segment {i+1}/{len(segments)}")

    inputs = tokenizer(
        segment,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=nepali_token_id,
            max_new_tokens=512,
            num_beams=5,
            do_sample=False,
        )

    translated_segment = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    translated_segments.append(translated_segment)


translated_text = "\n\n".join(translated_segments)

print(translated_text)