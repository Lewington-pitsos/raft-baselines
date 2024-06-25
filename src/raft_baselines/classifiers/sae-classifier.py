import numpy as np
from .classifier import Classifier
from typing import Callable, List, Mapping, Dict, Optional, Any
import datasets
from sklearn.tree import DecisionTreeClassifier
from torch import nn
import transformers
from transformer_lens import HookedTransformer


class SAEBaseModel(nn.Module):
    def __init__(self, 
                 transformer_name: str, 
                 sae_release: str,
                 sae_id:str,
                 device, 
                 max_seq_len:int = None, 
                 num_classes: int = 2,
        ):
        super(SAEBaseModel, self).__init__()

        sae, _, _ = SAE.from_pretrained(release = sae_release, sae_id = sae_id, device = device)

        self.sae = sae

        for param in self.sae.parameters():
            param.requires_grad = False

        seq_len = int(max_seq_len / 8)

        self.fc1 = nn.Linear(sae.cfg.d_sae * seq_len, num_classes, bias=False)
        self.device = device

        self.model = HookedTransformer.from_pretrained(transformer_name, device=device)
        self.hook_layer = sae.cfg.hook_layer
        self.hook_name = sae.cfg.hook_name

        for param in self.model.parameters():
            param.requires_grad = False


        self.dropout = nn.Dropout(0.3)
        self.avg_pool = nn.AvgPool1d(kernel_size=8, stride=8)


class SAEFeaturesModel(SAEBaseModel):
    def forward(self, input_ids, attention_mask):
        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True, stop_at_layer=self.hook_layer + 1)

        hidden_states = cache[self.hook_name]

        features = self.sae.encode(hidden_states)
        return features

def masked_avg(embedding_matrix, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1)
    
    sum_embedding = (embedding_matrix * attention_mask_expanded).sum(dim=1)
    non_masked_count = attention_mask.sum(dim=1, keepdim=True)
    
    non_masked_count = non_masked_count.clamp(min=1)
    
    average_embedding = sum_embedding / non_masked_count

    return average_embedding


class SAEClassifier(Classifier):
    def __init__(self, training_data: datasets.Dataset, model_name:str, device:Any, max_seq_len:int):
        super().__init__(training_data)
        if len(self.input_cols) != 1:
            raise ValueError("SAEClassifier only supports one input column, got", self.input_cols)
        self.text_col = self.input_cols[0]

        self.embedder = SAEFeaturesModel(
            device=device,
            max_seq_len=max_seq_len,
            transformer_name='gpt2',
            sae_release='gpt2-small-res-jb',
            sae_id='blocks.8.hook_resid_pre',
        )
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device  

        self.classifier = DecisionTreeClassifier()

        X = []
        y = []

        for sample in self.training_data:
            X.append(self._embed(sample))
            y.append(sample[self.class_col])

        X = np.array(X)
        y = np.array(y)

        self.classifier.fit(X, y)


    def _embed(self, sample):
        text = sample[self.text_col]
        input_ids, attention_mask = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.embedder.max_seq_len)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        sae_feature_sequence = self.embedder(input_ids=input_ids, attention_mask=attention_mask)

        avg_features = masked_avg(sae_feature_sequence, attention_mask)

        return avg_features

    def classify(self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
        should_print_prompt: bool = False,
        ):
        
        embedding = self.embed(target)
        prediction = self.classifier.predict(embedding)

        result = {c: 0.0 for c in self.classes}


        result[self.class_label_to_string(prediction[0])] = 1.0

        return result