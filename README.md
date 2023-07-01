# READY FOR TRAINING!!!!!!

# Agora
Agora is an new open source Multi-Modality AI Research Organization devoted to advancing Humanity!

Since Orca is ready to train Agora is actively seeking cloud providers or grant providers to train this all-new revolutionary model and release it open source, if you would like to learn more please email me at `kye@apac.ai`


![Agora banner](agora-banner.png)

[Join our Agora discord and contribute to this project or 40+ others!](https://discord.gg/qUtxnK2NMf)


# Orca: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

![Orca Next Generation Open Source Language Model](/orca-banner.png)

Orca is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Orca is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM.



# Usage
Get started:

1. Clone the repository and install the required packages.


```
git clone https://github.com/kyegomez/Orca
cd Orca
pip3 install -r requirements.txt
cd Orca
python3 training_distributed.py
```

# Training

First:

`Accelerate Config`

Enable Deepspeed 3: 

`Accelerate launch train_distributed_accelerate.py`



## Dataset building building

Data
You can preprocess a different dataset in a way similar to the C4 dataset used during training by running the build_dataset.py script. This will pre-tokenize, chunk the data in blocks of a specified sequence length, and upload to the Huggingface hub. For example:

```python3 Orca/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"```



# Inference

```python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "orca"``` 

Not yet we need to submit model to pytorch hub



## Model Architecture 🧠🔧

```python
model = TransformerWrapper(
        num_tokens=64007,
        max_seq_len=8192,
        use_abs_pos_emb=False,
        tokenizer=tokenizer, # !
        embedding_provider=AndromedaEmbedding(),
        attn_layers = Decoder(
            dim=128, # 2048
            depth=8, # 16
            dim_head=128,
            heads=8,
            alibi_pos_bias=True,
            alibi_num_heads=4,
            rotary_xpos=True,
            attn_flash = True,
            deepnorm=True,
            shift_tokens=1,
            attn_one_kv_head = True,
            qk_norm=True,
            attn_qk_norm=True,
            attn_qk_norm_dim_scale=True # set this to True, in addition to `attn_qk_norm = True`
        )
    )
```

## Roadmap 🗺️📍

1. **Training phase**: Train Orca on a large-scale dataset to achieve SOTA performance in various natural language processing tasks.

2. **World-class inference infrastructure**: Establish a robust and efficient infrastructure that leverages techniques such as:

   - Model quantization: Reduce memory and computational requirements without significant loss in performance.
   - Distillation: Train smaller, faster models that retain the knowledge of the larger model.
   - Optimized serving frameworks: Deploy Orca using efficient serving frameworks, such as NVIDIA Triton or TensorFlow Serving, for rapid inference.

3. **Continuous improvement**: Continuously fine-tune Orca on diverse data sources and adapt it to new tasks and domains.

4. **Community-driven development**: Encourage open-source contributions, including pre-processing improvements, advanced training techniques, and novel use cases.

## Why Orca? 🌠💡

Orca can potentially be finetuned with 100k+ token sequence length.
Orca is a state-of-the-art language model that leverages advanced techniques to optimize its performance and efficiency. Some of these techniques include alibi positional bias, rotary position encodings (xpos), flash attention, and deep normalization (deepnorm). Let's explore the benefits of these techniques and provide some usage examples.

### Alibi Positional Bias

Alibi positional bias allows the model to learn relative positions between tokens, enabling it to better capture the relationships and dependencies between tokens in a sequence.

Usage example:

```python
attn_layers = Decoder(
    ...
    alibi_pos_bias=True,
    alibi_num_heads=4,
    ...
)
```

### Rotary Position Encodings (xpos)

Rotary position encodings introduce a more efficient way to encode positions in the input sequence. They avoid the need for absolute positional embeddings, reducing the model's memory footprint and improving training speed.

Usage example:

```python
attn_layers = Decoder(
    ...
    rotary_xpos=True,
    ...
)
```

### Flash Attention

Flash attention speeds up the self-attention mechanism by reducing the number of attention computations. It accelerates training and inference while maintaining a high level of performance.

Usage example:

```python
attn_layers = Decoder(
    ...
    attn_flash=True,
    ...
)
```

Usage example:

```python
attn_layers = Decoder(
    ...
    deepnorm=True,
    ...
)
```

### Deep Normalization (deepnorm)

Deep normalization is a technique that normalizes the activations within a layer, helping with training stability and convergence. It allows the model to better learn complex patterns and generalize to unseen data.

# Orca Principles
- **Efficiency**: Orca incorporates cutting-edge optimization techniques, such as attention flashing, rotary position encodings, and deep normalization, resulting in efficient training and inference.

- **Flexibility**: The modular design of Orca allows for easy adaptation to various tasks and domains, making it a versatile choice for a wide range of applications.

- **Scalability**: Orca's architecture is designed to scale with the ever-growing computational resources and data sizes, ensuring its continuous relevance in the NLP landscape.

- **Community-driven**: As an open-source project, Orca thrives on contributions from the community, fostering an environment of collaboration, innovation, and continuous improvement.

Join us on this exciting journey to create a powerful, efficient, and intelligent language model that will revolutionize the NLP landscape! 🚀🌟

## Todo:

* Pretrain on Falcon

* [Finetune on this](https://huggingface.co/datasets/Open-Orca/OpenOrca)


# Training Orca Model

## Overview

This README provides step-by-step instructions on how to train the Orca model. This process encompasses aspects like tokenization, sequencing, and loss computation.

---

### Step 1: Tokenization

1. We use the LLaMA Byte Pair Encoding (BPE) tokenizer to process the input examples.
2. The LLaMA tokenizer splits all numbers into individual digits and decomposes unknown UTF-8 characters into bytes.
3. To handle sequences of variable length, we incorporate a padding token `[[PAD]]` into the LLaMA tokenizer vocabulary.
4. The final vocabulary comprises 32,001 tokens.

---

### Step 2: Packing

1. To optimize the training process, we utilize a packing technique. This technique concatenates multiple input examples into a single sequence for training the model.
2. We ensure that the total length of the concatenated sequence doesn't exceed `max_len=2048` tokens.
3. The input examples are shuffled, then partitioned into groups such that the length of the concatenated sequence in each group is at most `max_len`.
4. We then add padding tokens to the concatenated sequence to create a uniform input sequence length of `max_len`.
5. The average packing factor given the length distribution of the augmented instructions in our training data is 2.7 examples per sequence.

---

### Step 3: Loss Computation

1. For training Orca, the loss is computed only on the tokens generated by the teacher model. In other words, the model learns to generate responses based on the system message and task instructions.
2. This strategy ensures the model focuses on learning from the most pertinent and informative tokens, thereby improving the overall efficiency and effectiveness of the training process.

---

### Step 4: Compute Resources

1. We trained Orca on 20 NVIDIA A100 GPUs, each with 80GB memory.
2. The model was trained on FLAN-5M (ChatGPT augmentations) for 4 epochs over 160 hours, and on FLAN-1M (GPT-4 augmentations) for the same number of epochs over 40 hours.
3. Data collection from GPT-3.5-turbo (ChatGPT) and GPT-4 from multiple endpoints took 2 weeks and 3 weeks respectively, taking into account the throttling limit, endpoint load, and length distribution of query and response pairs.

---
