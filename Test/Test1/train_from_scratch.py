from transformers import (
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel,
    TextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)

def main():
    
    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=512,
        n_ctx=512,
        n_embd=768,    
        n_layer=12,    
        n_head=12      
    )

    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  

    
    model = GPT2LMHeadModel(config)

    
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train.txt",
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    
    training_args = TrainingArguments(
        output_dir="./scratch_gpt2",
        overwrite_output_dir=True,
        num_train_epochs=5,                
        per_device_train_batch_size=4,     
        save_steps=500,
        save_total_limit=2,
        logging_steps=100
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    
    trainer.train()
    trainer.save_model("./scratch_gpt2")
    tokenizer.save_pretrained("./scratch_gpt2")

if __name__ == "__main__":
    main()
