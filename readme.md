### Install

`Python>=3.9`

Requirements
```shell
python = "^3.9"
transformers = "^4.27.3"
datasets = "^2.10.1"
torch = "^2.0.0"
scikit-learn = "^1.2.2"
```

Training Command. I ran on 8x 48GB NVIDIA RTX A6000.

```shell
python train.py -m "google/flan-t5-large" --train --overwrite-cache -b 2 --accum 4
```

### Performance

After 500 training steps, this model scored 82.2% on the test data.

### HuggingFace

Please find the model on the huggingface hub [here](https://huggingface.co/dslack/Grade-School-Math-Flan-T5-Large).

### Generation

Here is how to generate with the model, here with nucleus sampling.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("dslack/Grade-School-Math-Flan-T5-Large")
model = AutoModelForSeq2SeqLM.from_pretrained("dslack/Grade-School-Math-Flan-T5-Large")

text = "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"
encoded_text = tokenizer.encode(text, return_tensors="pt").cuda()
generation = model.generate(encoded_text, do_sample=True, top_p=0.9, max_length=512).cpu()
result = tokenizer.decode(generation[0], skip_special_tokens=True)
print(result)
```

which gives us

```shell
20% of the students are enrolled in contemporary dance, so 20 x 20/100 = 8 students are not enrolled in contemporary dance. Therefore, 20 - 8 = 14 students are not enrolled in jazz dance. Therefore, 14 x 25/100 = 5 students are not enrolled in jazz dance. Therefore, 14 - 5 = 6 students are not enrolled in hip-hop dance. Therefore, 6 x 100 = 60% of the students are not enrolled in hip-hop dance. The answer is 60
```

Another example
```
Input: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
Output: First find how many bolts of white fiber it takes: 2 bolts/half = 1 bolt/bolt Then add that amount to the blue fiber to find the total number of bolts: 1 bolt + 2 bolts = 3 bolts The answer is 3
```
