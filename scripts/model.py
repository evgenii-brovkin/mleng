import time
import logging

import fire


def main():
    fire.Fire(TrainModel)


class TrainModel:
    def train(self, prepared_data_path, save_model_path):
        logger.info(f"Loading data from {prepared_data_path}")
        time.sleep(0.5)
        logger.info(f"Training model and saving to {save_model_path}")
        time.sleep(0.5)

    def tune(self, prepared_data_path):
        logger.info(f"Loading data from {prepared_data_path}")
        time.sleep(0.5)
        logger.info("Tunning hyperparameters")
        time.sleep(0.5)
        logger.info("Logging results to mlflow")
        time.sleep(0.5)

    def evaluate(self, prepared_data_path, model_output_path):
        logger.info(f"Loading data from {prepared_data_path}")
        time.sleep(0.5)
        logger.info("Evaluating data")
        time.sleep(0.5)
        logger.info(f"Saving results to {model_output_path}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

if __name__ == "__main__":
    main()
