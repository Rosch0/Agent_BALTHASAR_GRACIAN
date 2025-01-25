from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch import nn

class FalconModelTrainer:
    def __init__(self, model_name, data_path, output_dir):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Add padding token to tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and preprocess dataset
        self.dataset = self._load_dataset()
        self.tokenized_datasets = self._tokenize_dataset()

        # Initialize training arguments
        self.training_args = self._set_training_arguments()

    def _load_dataset(self):
        dataset = load_dataset("json", data_files=self.data_path)
        return dataset["train"].train_test_split(test_size=0.1)

    def _tokenize_dataset(self):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        return self.dataset.map(tokenize_function, batched=True)

    def _set_training_arguments(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            eval_steps=1000,
            learning_rate=3e-4,
            warmup_steps=500,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=500,
            dataloader_num_workers=0,
            push_to_hub=False
        )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["input_ids"]

            # Shift labels and logits for causal language modeling
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    def train_model(self):
        trainer = self.CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            tokenizer=self.tokenizer
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

if __name__ == "__main__":
    model_name = "tiiuae/falcon-rw-1b"
    json_data_path = r"C:\Users\Blaze\Desktop\AgentAI\Gracian_cleaned_data2.json"
    output_dir = "./falcon-finetunedXXX"

    trainer = FalconModelTrainer(model_name, json_data_path, output_dir)
    trainer.train_model()
