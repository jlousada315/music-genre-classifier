import argparse
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate

class AudioClassifier:
    def __init__(self, model_id, dataset_name, max_duration=30.0):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.max_duration = max_duration

        # Load dataset
        self.dataset = self.load_dataset()
        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id, num_labels=len(self.dataset['train'].features['label'].names))

    def load_dataset(self):
        """Load and preprocess the dataset."""
        dataset = load_dataset(self.dataset_name)
        sampling_rate = self.feature_extractor.sampling_rate
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        return dataset

    def preprocess_function(self, examples):
        """Preprocess the audio examples for the model."""
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
            return_attention_mask=True,
        )
        return inputs

    def train(self, training_args):
        """Train the model."""
        encoded_dataset = self.dataset.map(
            self.preprocess_function,
            remove_columns=["audio", "file"],
            batched=True,
            batch_size=100,
            num_proc=1,
            keep_in_memory=True
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(predictions=predictions, references=eval_pred.label_ids)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            compute_metrics=compute_metrics
        )

        trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Audio Classification")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID from Hugging Face Hub.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for model checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    args = parser.parse_args()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Create an instance of AudioClassifier
    classifier = AudioClassifier(model_id=args.model_id, dataset_name=args.dataset_name)
    classifier.train(training_args)

if __name__ == "__main__":
    main()
