import numpy as np
import gradio as gr
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from huggingface_hub import notebook_login
import evaluate

class MusicGenreClassifier:
    def __init__(self, model_id):
        self.model_id = model_id
        self.dataset = None
        self.feature_extractor = None
        self.model = None
        self.trainer = None

    def load_dataset(self):
        self.dataset = load_dataset("marsyas/gtzan", "all", trust_remote_code=True)
        self.dataset = self.dataset["train"].train_test_split(seed=42, test_size=0.1)

    def initialize_feature_extractor(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)

    def preprocess_data(self):
        sampling_rate = self.feature_extractor.sampling_rate
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

        def preprocess_function(examples):
            audio_arrays = [x["array"] for x in examples["audio"]]
            return self.feature_extractor(audio_arrays, sampling_rate=sampling_rate, return_attention_mask=True)

        self.dataset = self.dataset.map(preprocess_function, remove_columns=["audio", "file"], batched=True)
        self.dataset = self.dataset.rename_column("genre", "label")

    def initialize_model(self):
        num_labels = len(self.dataset["train"].features["label"].names)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id, num_labels=num_labels)

    def train_model(self):
        notebook_login()
        training_args = TrainingArguments(
            f"{self.model_id.split('/')[-1]}-finetuned-gtzan",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=5,
            num_train_epochs=10,
            logging_steps=5,
            load_best_model_at_end=True,
            push_to_hub=True
        )

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(predictions=predictions, references=eval_pred.label_ids)

        self.trainer = Trainer(
            self.model,
            training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            compute_metrics=compute_metrics
        )
        self.trainer.train()

    def generate_audio(self):
        example = self.dataset["train"].shuffle()[0]
        audio = example["audio"]
        return audio["sampling_rate"], audio["array"], self.dataset["train"].features["label"].int2str(example["genre"])

    def launch_gradio_interface(self):
        with gr.Blocks() as demo:
            with gr.Column():
                for _ in range(4):
                    audio, label = self.generate_audio()
                    gr.Audio(audio, label=label)

        demo.launch(debug=True)

if __name__ == "__main__":
    model_id = "ntu-spml/distilhubert"
    classifier = MusicGenreClassifier(model_id)
    classifier.load_dataset()
    classifier.initialize_feature_extractor()
    classifier.preprocess_data()
    classifier.initialize_model()
    classifier.train_model()
    classifier.launch_gradio_interface()
