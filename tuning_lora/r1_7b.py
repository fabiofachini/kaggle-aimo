# Importações necessárias
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
import pandas as pd
import os
import wandb

# Inicialização do Weights & Biases para monitoramento do treinamento
wandb.init(
    project="fine-tuning-llm",  # Nome do projeto no W&B
    name="deepseek_lora_finetune",  # Nome da execução do experimento
    config={  # Hiperparâmetros do treinamento
        "model": "DeepSeek-R1-Distill-Qwen-7B",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "epochs": 3,
        "learning_rate": 5e-5,
        "precision": "fp16",
        "adapter": "LoRA"
    }
)

# Caminho do modelo pré-treinado
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Carregar o modelo pré-treinado sem quantização
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Carregar o tokenizador correspondente ao modelo
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Carregar o dataset a partir de um arquivo CSV
dataset_path = "final_simple_df.csv"
df = pd.read_csv(dataset_path)

# Converter a coluna 'extracted_answer' para numérica, tratando erros como NaN
df.loc[:, 'extracted_answer'] = pd.to_numeric(df['extracted_answer'], errors='coerce')

# Remover linhas com valores NaN nas colunas 'awnser' ou 'extracted_answer'
df = df.dropna(subset=['awnser', 'extracted_answer'])

# Converter as colunas 'awnser' e 'extracted_answer' para inteiros
df.loc[:, 'awnser'] = df['awnser'].astype(int)
df.loc[:, 'extracted_answer'] = df['extracted_answer'].astype(int)

# Filtrar o dataset para manter apenas as linhas onde 'awnser' e 'extracted_answer' são iguais
filtered_df = df[df["awnser"] == df["extracted_answer"]]

# Manter apenas a coluna 'message' no dataset filtrado
filtered_df = filtered_df[["message"]]

# Verificar o resultado do dataset filtrado
print(filtered_df.head())  # Exibe as primeiras linhas do dataset filtrado

# Converter o DataFrame filtrado para o formato do Hugging Face Dataset
filtered_dataset = Dataset.from_pandas(filtered_df)

# Função para formatar os exemplos do dataset
def format_example(example):
    return {"text": example["message"]}  # Usa diretamente a coluna "message"

# Aplicar a função de formatação ao dataset
formatted_data = filtered_dataset.map(format_example)

# Função para tokenizar os exemplos do dataset
def tokenize_function(example):
    return tokenizer(example["text"],
                     padding="max_length",  # Garante que todos os exemplos tenham o mesmo tamanho.
                     truncation=False,      # Corta textos muito longos.
                     max_length=12288)      # Limita o tamanho da entrada para evitar estouro de memória.

# Aplicar a função de tokenização ao dataset formatado
tokenized_datasets = formatted_data.map(tokenize_function, batched=True)

# Configuração do LoRA (Low-Rank Adaptation)
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                                # Tamanho dos adaptadores LoRA
    lora_alpha=16,                       # Escalabilidade do aprendizado
    lora_dropout=0.05,                   # Previne overfitting
    target_modules=["q_proj", "v_proj"] # Ajusta apenas partes específicas do modelo
)

# Aplicar LoRA ao modelo
model = get_peft_model(model, config)

# Configuração dos argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Tamanho do batch por dispositivo
    gradient_accumulation_steps=4,  # Número de passos de acumulação de gradiente
    num_train_epochs=3,             # Número de épocas de treinamento
    save_strategy="epoch",          # Salvar o modelo a cada época
    logging_dir="./logs",           # Diretório para logs
    report_to="wandb",              # Relatar métricas para o W&B
    fp16=True,                      # Usar precisão reduzida (fp16) para treinar mais rápido
    learning_rate=5e-5,             # Taxa de aprendizado
)

# Inicializar o Trainer para treinar o modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Iniciar o treinamento
trainer.train()

# Salvar o modelo e o tokenizador após o treinamento
output_model_path = "./fine_tuned_model"
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)

# Mensagem de sucesso
print("\u2705 Modelo salvo com sucesso!")

# Finalizar a execução no W&B
wandb.finish()
