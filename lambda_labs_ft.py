import lambda_labs
import os
import textwrap

# --- Configuration ---
# Set the name of the SSH key you added to your Lambda Labs account.
# This is a mandatory step.
SSH_KEY_NAME = "YOUR_SSH_KEY_NAME" 

# Choose the GPU instance type. For Llama 4 Maverick (~400B), an H100 or A100 is best.
# For Scout (~109B), a lower-tier card might work, but an A100 is a safe bet.
# Find instance names by running `lambda-labs instances` in your terminal.
INSTANCE_TYPE = "a100.80gb.pcie" 

# --- Startup Script (cloud-init) ---
# This is a bash script that will be executed on the new instance upon boot.
# It sets up the environment, writes the Python fine-tuning code to a file,
# runs the training, and then terminates the instance.

user_data_script = textwrap.dedent("""\
    #!/bin/bash
    
    # 1. SETUP: Update, install git, and set up the Python environment
    # ---------------------------------------------------------------
    apt-get update
    apt-get install -y git python3-pip
    
    # Clone the transformers repo for any potential utilities (optional but good practice)
    git clone https://github.com/huggingface/transformers.git
    
    # Install all required Python packages
    pip install -q -U transformers datasets accelerate peft bitsandbytes trl torch
    
    # 2. PYTHON SCRIPT: Write our fine-tuning code to a file
    # --------------------------------------------------------
    # Using cat and EOF to write the multi-line Python script into fine_tune.py
    cat > /root/fine_tune.py << 'EOF'
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer
    import os

    print("--- Starting Fine-Tuning Job ---")

    # Configuration for the fine-tuning job
    MODEL_ID = "meta-llama/Llama-4-8B-chat-hf" # Using a smaller Llama 4 for demonstration
    DATASET_ID = "AlicanKiraz0/All-CVE-Records-Training-Dataset"
    NEW_MODEL_NAME = "Llama-4-8B-CVE-Solutions-Specialist"

    # Load and prepare the dataset
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")

    def format_prompt(example):
        return {"text": f"### INSTRUCTION:\\nYou are a cybersecurity expert. Based on the following CVE description, provide a detailed solution or mitigation.\\n\\n### CVE Description:\\n{example['Description']}\\n\\n### SOLUTION:\\n{example['Solution']}"}

    dataset = dataset.map(format_prompt)
    print("Dataset prepared successfully.")

    # Configure quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer and model
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    print("Model configured with QLoRA.")

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=f"/root/{NEW_MODEL_NAME}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        group_by_length=True,
    )

    # Create and run the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
    )

    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")
    
    print(f"Saving fine-tuned model to /root/{NEW_MODEL_NAME}")
    trainer.save_model(f"/root/{NEW_MODEL_NAME}")
    print("--- Model Saved ---")
    
    # IMPORTANT: Add code here to upload your model to S3, Hugging Face Hub, etc.
    # Example: aws s3 sync /root/Llama-4-8B-CVE-Solutions-Specialist s3://my-models-bucket/
    
    EOF

    # 3. EXECUTION: Run the Python script we just created
    # ----------------------------------------------------
    python3 /root/fine_tune.py
    
    # 4. CLEANUP: Terminate the instance to prevent further charges
    # -------------------------------------------------------------
    echo "Fine-tuning job complete. Terminating instance."
    # The instance will be terminated by an API call from the controller script.
    # As a fallback, you can use: shutdown now
    
""")

# --- API Call to Launch Instance ---
if __name__ == "__main__":
    if "LAMBDA_API_KEY" not in os.environ:
        print("Error: LAMBDA_API_KEY environment variable not set.")
        exit(1)
        
    if "YOUR_SSH_KEY_NAME" in SSH_KEY_NAME:
        print("Error: Please replace 'YOUR_SSH_KEY_NAME' with your actual SSH key name.")
        exit(1)

    print(f"Attempting to launch a '{INSTANCE_TYPE}' instance...")
    print(f"It will be provisioned with SSH key: '{SSH_KEY_NAME}'")

    try:
        # Launch the instance with the startup script
        instance_id = lambda_labs.launch(
            instance_type_name=INSTANCE_TYPE,
            ssh_key_names=[SSH_KEY_NAME],
            user_data=user_data_script,
        )[0] # launch returns a list of launched instance IDs

        print("\n🚀 Instance launched successfully!")
        print(f"Instance ID: {instance_id}")
        print("The instance will now boot, install dependencies, and run the fine-tuning job.")
        print("You can monitor its progress by SSHing into it or checking the logs in the Lambda Labs dashboard.")
        
        # You can add logic here to monitor the instance and terminate it
        # once the job is done, but the startup script handles termination as a fallback.
        
    except Exception as e:
        print(f"An error occurred: {e}")
