import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from langchain import LangChain

# Load BERT Tiny model and tokenizer
model_name = "VenkatManda/bert-tiny-squadV2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Initialize LangChain for context management
lang_chain = LangChain()

# Streamlit UI
st.title("Question Answering with BERT Tiny and LangChain")

# Input box for user context
context = st.text_area("Enter your context:")

# Input box for user question
question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    with lang_chain("en"):
        # Tokenize input and convert to torch tensors
        inputs = tokenizer(question, context, return_tensors="pt")
        # Get model output
        outputs = model(**inputs)
        # Extract start and end logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the answer span
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits) + 1
        answer = tokenizer.decode(inputs["input_ids"][0, start_index:end_index])

        # Display the answer
        st.success(f"Answer: {answer}")
