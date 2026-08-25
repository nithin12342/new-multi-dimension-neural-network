---
license: mit
---

# ERNIE-Layout_Pytorch

- **Model type:** [ERNIE-Layout](https://arxiv.org/abs/2210.06155)
- **Repository:** [source code](https://github.com/NormXU/ERNIE-Layout-Pytorch): an unofficial ERNIE-Layout implementation in Pytorch
  
- **Converted from:** [PaddlePaddle/ernie-layoutx-base-uncased](https://huggingface.co/PaddlePaddle/ernie-layoutx-base-uncased)
  

The ERNIE-Layout-Pytorch model is initially released by [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP). To make Pytorch users easy to use, the model has been converted into PyTorch format with the [tools/convert2torch.py](https://github.com/NormXU/ERNIE-Layout-Pytorch/blob/main/tools/convert2torch.py) script.
Please feel free to make any changes you need. For more details and use cases, please check the repo.

**A Quick Example**
```python
import torch
from PIL import Image
import torch.nn.functional as F
from networks import ErnieLayoutConfig, ErnieLayoutForQuestionAnswering, \
    ErnieLayoutProcessor, ErnieLayoutTokenizerFast
from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor

pretrain_torch_model_or_path = "Norm/ERNIE-Layout-Pytorch"
doc_imag_path = "./dummy_input.jpeg"

context = ['This is an example sequence', 'All ocr boxes are inserted into this list']
layout = [[381, 91, 505, 115], [738, 96, 804, 122]]  # make sure  all boxes are normalized between 0 - 1000
pil_image = Image.open(doc_imag_path).convert("RGB")

# initialize tokenizer
tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)

# initialize feature extractor
feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
processor = ErnieLayoutProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

# Tokenize context & questions
question = "what is it?"
encoding = processor(pil_image, question, context, boxes=layout, return_tensors="pt")

# dummy answer start && end index
start_positions = torch.tensor([6])
end_positions = torch.tensor([12])

# initialize config
config = ErnieLayoutConfig.from_pretrained(pretrained_model_name_or_path=pretrain_torch_model_or_path)
config.num_classes = 2  # start and end

# initialize ERNIE for VQA
model = ErnieLayoutForQuestionAnswering.from_pretrained(
    pretrained_model_name_or_path=pretrain_torch_model_or_path,
    config=config,
)

output = model(**encoding, start_positions=start_positions, end_positions=end_positions)

# decode output
start_max = torch.argmax(F.softmax(output.start_logits, dim=-1))
end_max = torch.argmax(F.softmax(output.end_logits, dim=-1)) + 1  # add one ##because of python list indexing
answer = tokenizer.decode(encoding.input_ids[0][start_max: end_max])
print(answer)


```