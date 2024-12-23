import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
from typing import Union, List

class CLIPTextEmbedding:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        local_model_path = "/home/zhaoyixiu/ISR_project/CLIP"

        self.model = CLIPModel.from_pretrained(local_model_path + '/model').to(self.device)
        # self.processor = CLIPProcessor.from_pretrained(model_name)
        self.processor = CLIPTokenizer.from_pretrained(local_model_path + '/tokenizer')
        
        # local_model_path = "/home/zhaoyixiu/ISR_project/CLIP"
        # self.model.save_pretrained(local_model_path + '/model')
        # self.processor.save_pretrained(local_model_path + '/tokenizer')
        
        print("save model ok!")
        
        self.model.eval()
        
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        # if isinstance(text, str):
            # text = [text]
            
        with torch.no_grad():
            
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            
            print(f"Vocabulary size: {self.processor.vocab_size}")
            print(f"Max token ID in inputs: {inputs.input_ids.max()}")
            print(f"Min token ID in inputs: {inputs.input_ids.min()}")
            
            text_features = self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()})
            # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print("shapef of inputs: ", inputs['input_ids'].shape, inputs['attention_mask'].shape)
            # inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
            # inputs['attention_mask'] = inputs["attention_mask"].unsqueeze(0)
            
            # text_features = self.model.get_text_features(**inputs)
            
            if normalize:
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            print("shape of text_features:", text_features.shape)
                
            embeddings = text_features.cpu().numpy()
            
        return embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        return np.dot(emb1, emb2.T)[0][0]

# Example usage
def main():
    clip_encoder = CLIPTextEmbedding()
    
    # Single text example
    embedding = clip_encoder.encode_text("A beautiful sunset")
    print(f"Embedding shape: {embedding.shape}")
    
    # Similarity example
    similarity = clip_encoder.calculate_similarity(
        "A dog running", 
        "A puppy playing"
    )
    print(f"Similarity score: {similarity:.3f}")

if __name__ == "__main__":
    main()