import os
from google.cloud import storage

gcp_project = os.environ["GCP_PROJECT"]
bucket_name = "ac215-project"
persistent_folder = "/persistent"
image_path = os.path.join(persistent_folder, "image")
cgp_model_folder_transformer = "transformer_model"
transformer_model_path = os.path.join(persistent_folder, cgp_model_folder_transformer)
cgp_model_folder_prefix = "prefix_model"
prefix_model_path = os.path.join(persistent_folder, cgp_model_folder_prefix)
cgp_model_folder_distill = "distilled_prefix_model"
distill_model_path = os.path.join(persistent_folder, cgp_model_folder_distill)
cgp_model_folder_rnn = "rnn_model"
rnn_model_path = os.path.join(persistent_folder, cgp_model_folder_rnn)

if not os.path.exists(image_path):
    os.mkdir(image_path)
if not os.path.exists(transformer_model_path):
    os.mkdir(transformer_model_path)
if not os.path.exists(prefix_model_path):
    os.mkdir(prefix_model_path)
if not os.path.exists(distill_model_path):
    os.mkdir(distill_model_path)
if not os.path.exists(rnn_model_path):
    os.mkdir(rnn_model_path)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=gcp_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def download_test_image():
    test_image_file = "test_image.jpeg"

    if not os.path.exists(os.path.join(image_path, test_image_file)):
        print('Downloading test image...')
        download_blob(bucket_name, test_image_file, os.path.join(image_path, test_image_file))


def download_transformer_model():
    vectorization_file = os.path.join(cgp_model_folder_transformer,"vectorization_weights.pkl")
    encoder_file = os.path.join(cgp_model_folder_transformer,"encoder.h5")
    decoder_file = os.path.join(cgp_model_folder_transformer,"decoder.h5")

    if not os.path.exists(os.path.join(persistent_folder, vectorization_file)):
        print('Downloading transformer_model vectorization...')
        download_blob(bucket_name, vectorization_file, os.path.join(persistent_folder, vectorization_file))

    if not os.path.exists(os.path.join(persistent_folder, encoder_file)):
        print('Downloading transformer_model encoder weights...')
        download_blob(bucket_name, encoder_file, os.path.join(persistent_folder, encoder_file))

    if not os.path.exists(os.path.join(persistent_folder, decoder_file)):
        print('Downloading transformer_model decoder weights...')
        download_blob(bucket_name, decoder_file, os.path.join(persistent_folder, decoder_file))

    print('Done!')

def download_prefix_model():
    vectorization_file = os.path.join(cgp_model_folder_prefix,"vectorization_weights.pkl")
    encoder_file = os.path.join(cgp_model_folder_prefix,"encoder.h5")
    decoder_file = os.path.join(cgp_model_folder_prefix,"decoder.h5")

    if not os.path.exists(os.path.join(persistent_folder, vectorization_file)):
        print('Downloading prefix_model vectorization...')
        download_blob(bucket_name, vectorization_file, os.path.join(persistent_folder, vectorization_file))

    if not os.path.exists(os.path.join(persistent_folder, encoder_file)):
        print('Downloading prefix_model encoder weights...')
        download_blob(bucket_name, encoder_file, os.path.join(persistent_folder, encoder_file))

    if not os.path.exists(os.path.join(persistent_folder, decoder_file)):
        print('Downloading prefix_model decoder weights...')
        download_blob(bucket_name, decoder_file, os.path.join(persistent_folder, decoder_file))

    print('Done!')

def download_distilled_prefix_model():
    vectorization_file = os.path.join(cgp_model_folder_distill,"vectorization_weights.pkl")
    encoder_file = os.path.join(cgp_model_folder_distill,"encoder.h5")
    decoder_file = os.path.join(cgp_model_folder_distill,"decoder.h5")

    if not os.path.exists(os.path.join(persistent_folder, vectorization_file)):
        print('Downloading distilled_prefix_model vectorization...')
        download_blob(bucket_name, vectorization_file, os.path.join(persistent_folder, vectorization_file))

    if not os.path.exists(os.path.join(persistent_folder, encoder_file)):
        print('Downloading distilled_prefix_model encoder weights...')
        download_blob(bucket_name, encoder_file, os.path.join(persistent_folder, encoder_file))

    if not os.path.exists(os.path.join(persistent_folder, decoder_file)):
        print('Downloading distilled_prefix_model decoder weights...')
        download_blob(bucket_name, decoder_file, os.path.join(persistent_folder, decoder_file))

    print('Done!')

def download_rnn_model():
    vectorization_file = os.path.join(cgp_model_folder_rnn,"tokenizer.json")
    encoder_file = os.path.join(cgp_model_folder_rnn,"encoder.h5")
    decoder_file = os.path.join(cgp_model_folder_rnn,"decoder.h5")

    if not os.path.exists(os.path.join(persistent_folder, vectorization_file)):
        print('Downloading rnn_model vectorization...')
        download_blob(bucket_name, vectorization_file, os.path.join(persistent_folder, vectorization_file))

    if not os.path.exists(os.path.join(persistent_folder, encoder_file)):
        print('Downloading rnn_model encoder weights...')
        download_blob(bucket_name, encoder_file, os.path.join(persistent_folder, encoder_file))

    if not os.path.exists(os.path.join(persistent_folder, decoder_file)):
        print('Downloading rnn_model decoder weights...')
        download_blob(bucket_name, decoder_file, os.path.join(persistent_folder, decoder_file))

    print('Done!')

if __name__ == "__main__":
    download_test_image()
    download_transformer_model()
    download_prefix_model()
    download_distilled_prefix_model()
    download_rnn_model()