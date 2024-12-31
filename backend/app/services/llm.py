from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
from ..config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(settings.MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Store conversation history
        self.conversation_history = []

    def _generate_response(self, prompt: str) -> str:
        """
        Generate response using the model
        """
        # Tokenize the input
        inputs = self.tokenizer(
            prompt,
            max_length=settings.MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate the output
        outputs = self.model.generate(
            **inputs,
            max_length=settings.MAX_LENGTH,
            num_return_sequences=1,
            temperature=settings.TEMPERATURE,
            top_k=settings.TOP_K,
            top_p=settings.TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def get_response(self, query: str, relevant_docs: list[str]) -> str:
        # Combine context and create prompt
        context = "\n".join(relevant_docs)
        prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.

Now use the following context items to answer the user query:
{context}

Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

        # Store conversation in history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Store response in history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []