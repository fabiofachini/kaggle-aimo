# Importações necessárias
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi, login
import torch
import pandas as pd
import wandb

# 🔹 Autenticação no Hugging Face
login()

# 🔹 Inicialização do Weights & Biases para monitoramento do treinamento
wandb.init(
    project="fine-tuning-llm",  
    name="deepseek_lora_finetune",  
    config={  
        "model": "DeepSeek-R1-Distill-Qwen-7B",
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "learning_rate": 5e-5,
        "precision": "fp16",
        "adapter": "LoRA"
    }
)

# 🔹 Caminho do modelo pré-treinado
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 🔹 Carregar o modelo e o tokenizador
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 🔹 Carregar dataset
dataset_path = "final_simple_df.csv"
df = pd.read_csv(dataset_path)

# 🔹 Limpeza e preparação dos dados
df['extracted_answer'] = pd.to_numeric(df['extracted_answer'], errors='coerce')
df = df.dropna(subset=['awnser', 'extracted_answer'])
df['awnser'] = df['awnser'].astype(int)
df['extracted_answer'] = df['extracted_answer'].astype(int)
filtered_df = df[df["awnser"] == df["extracted_answer"]][["message"]]

# 🔹 Converter para dataset Hugging Face
filtered_dataset = Dataset.from_pandas(filtered_df)
formatted_data = filtered_dataset.map(lambda x: {"text": x["message"]})

# 🔹 Tokenização dos dados
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=False, max_length=12288)

tokenized_datasets = formatted_data.map(tokenize_function, batched=True)

# 🔹 Configuração do LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                                
    lora_alpha=16,                     
    lora_dropout=0.05,                  
    target_modules=["q_proj", "v_proj"]
)

# 🔹 Aplicar LoRA ao modelo
model = get_peft_model(model, config)

# 🔹 Configuração dos argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="wandb",
    fp16=True,
    learning_rate=5e-5,
)

# 🔹 Inicializar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# 🔹 Iniciar o treinamento
trainer.train()

# 🔹 Caminho de saída do modelo treinado
output_model_path = "./fine_tuned_model"

# 🔹 Salvar modelo localmente
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)

# ✅ Mensagem de sucesso
print("\u2705 Modelo salvo localmente!")

# 🔹 Envio para o Hugging Face 🚀
repo_name = "fabiofachini/DeepSeek-R1-Distill-Qwen-1.5B-fabio"  # Nome do repositório no Hugging Face

# Fazer upload do modelo e tokenizador
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"✅ Modelo enviado para Hugging Face: https://huggingface.co/{repo_name}")

# 🔹 Finalizar o Weights & Biases
wandb.finish()
