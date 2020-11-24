import time
import logging

import fire


def main():
    fire.Fire(FeaturePreparation)


class FeaturePreparation:
    def ingest(self, raw_data_path, ingested_data_path):
        logger.info(f"Fetching data from {raw_data_path}")
        time.sleep(0.5)
        logger.info(f"Saceing data to {ingested_data_path}")
        time.sleep(0.5)

    def split(self, ingested_data_path, splitted_data_path):
        logger.info(f"Loading data from {ingested_data_path}")
        time.sleep(0.5)
        logger.info(f"Splitting data into {splitted_data_path}/train and {splitted_data_path}/test")
        time.sleep(0.5)

    def prepare(self, splitted_data_path, prepared_data_path):
        logger.info(f"Loading data from {splitted_data_path}")
        time.sleep(0.5)
        logger.info("Preparing features")
        time.sleep(0.5)
        logger.info(
            f"Saving into processed data into {prepared_data_path}/train and {prepared_data_path}/test"
        )
        time.sleep(0.5)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("features")

if __name__ == "__main__":
    main()
