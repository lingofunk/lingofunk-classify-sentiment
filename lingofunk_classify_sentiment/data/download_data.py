import hashlib
import logging
import os
import subprocess
import sys
import zipfile
from argparse import ArgumentParser

import requests

from lingofunk_classify_sentiment.config import config, fetch

logger = logging.getLogger(__name__)


def download_embedding(embedding):
    settings = config["embeddings"][embedding]

    glove_basepath = fetch(settings["basepath"])
    glove_zip_path = f"{glove_basepath}.zip"
    glove_unzip_path = f"{glove_basepath}.txt"

    glove_url = settings["url"]

    # Download the GloVe data if applicable
    if os.path.isfile(glove_zip_path):
        logger.info("GloVe data already exists, skipping download.")
    else:
        logger.info("Downloading GloVe data to {}".format(glove_zip_path))
        try:
            args = ["wget", "-O", glove_zip_path, glove_url]
            output = subprocess.Popen(args, stdout=subprocess.PIPE)
            out, err = output.communicate()
        except:
            logger.info(
                "Couldn't download GloVe data with wget, "
                "falling back to (slower) Python downloading."
            )
            glove_response = requests.get(glove_url, stream=True)
            with open(glove_zip_path, "wb") as glove_file:
                for chunk in glove_response.iter_content(chunk_size=1024 * 1024):
                    # Filter out keep-alive new chunks.
                    if chunk:
                        glove_file.write(chunk)

    # Extract the GloVe data if it does not already exist.
    if os.path.exists(glove_unzip_path):
        logger.info("Unzipped GloVe data already exists, skipping unzip.")
    else:
        logger.info("Unzipping GloVe archive to {}".format(glove_unzip_path))
        zip_ref = zipfile.ZipFile(glove_zip_path, "r")
        zip_ref.extractall(os.path.dirname(glove_unzip_path))
        zip_ref.close()


if __name__ == "__main__":
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s " "- %(name)s - %(message)s",
        level=logging.INFO,
        handlers=handlers,
    )
    parser = ArgumentParser(
        description=("Download data used in phonesthemes extraction.")
    )
    parser.add_argument(
        "--name",
        action="store",
        dest="name",
        type=str,
        help="Name of the embedding listed in config.json to download.",
    )

    args = parser.parse_args()

    if args.name:
        download_embedding(args.name)
    else:
        logger.info("No embedding name given, aborting.")
        sys.exit(2)
