from datasets import load_from_disk, load_metric
from textSummarizer.entity import ModelEvaluationConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]


    def calculate_metric_on_test_ds(self, dataset, batch_size=16, column_text="dialogue", column_summary="summary"):
        """create the batches of data for dialogue and summary for the given batch size
        """
        device=self.device
        dialogue_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        summary_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        return dialogue_batches, summary_batches


    def compute_metric(self, metric, model, tokenizer, dataset, device):

        dialogue_batches, summary_batches = self.calculate_metric_on_test_ds(dataset, batch_size = 2,
                                                                    column_text = "dialogue", column_summary = "summary")

        for dialogue_batch, summary_batch in tqdm(zip(dialogue_batches, summary_batches), total=len(dialogue_batches)):

            # here we are getting input_id and attention_mask
            inputs = tokenizer(dialogue_batch, max_length=1024,  truncation=True, padding="max_length", return_tensors="pt")

            # here we are generating summaries based on the inputs
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            length_penalty=0.8, num_beams=8, max_length=128)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

            # Finally, we decode the generated texts,
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=summary_batch)

        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score


    def evaluate(self):
        device = self.device
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        # loading data
        dataset_samsum = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load_metric('rouge')

        score = self.compute_metric(rouge_metric, model_pegasus, tokenizer, dataset_samsum['test'][0:10], device)

        rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

        df = pd.DataFrame(rouge_dict, index = [f'pegasus'] )
        df.to_csv(self.config.metric_file_name, index=False)