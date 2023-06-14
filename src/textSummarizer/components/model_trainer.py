from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model_pegasus)

        # loading_data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # provide the training arguments
        trainer_args = TrainingArguments(
             output_dir = self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
             per_device_train_batch_size=self.config.per_device_train_batch_size,
             weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps, save_steps=self.config.save_steps,
             gradient_accumulation_steps=self.config.gradient_accumulation_steps
            )

        # Here we are performing the training, also we are taking test data for my training because train data size is huge
        trainer = Trainer(model=model_pegasus, args=trainer_args, tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt['test'], eval_dataset=dataset_samsum_pt['validation'])
        
        trainer.train()

        # Save model
        model_pegasus.save_petrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))

        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

        # In above while performing Trainer(...) we can take train in our case but for explaining purpose we
        # have taken test dataset