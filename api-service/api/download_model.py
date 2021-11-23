import os
from google.cloud import storage

gcp_project = os.environ["GCP_PROJECT"]
bucket_name = "ac215-project"
persistent_folder = "/persistent"
cgp_model_folder_transformer = "transformer_model"
transformer_model_path = os.path.join(persistent_folder, cgp_model_folder_transformer)
image_path = os.path.join(persistent_folder, "image")

if not os.path.exists(transformer_model_path):
    os.mkdir(transformer_model_path)
if not os.path.exists(image_path):
    os.mkdir(image_path)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=gcp_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def download_transformer_model():
    vectorization_file = os.path.join(cgp_model_folder_transformer,"vectorization_weights.pkl")
    encoder_file = os.path.join(cgp_model_folder_transformer,"encoder.h5")
    decoder_file = os.path.join(cgp_model_folder_transformer,"decoder.h5")
    test_image_file = "test_image.jpeg"

    if not os.path.exists(os.path.join(persistent_folder, vectorization_file)):
        print('Downloading vectorization...')
        download_blob(bucket_name, vectorization_file, os.path.join(persistent_folder, vectorization_file))

    if not os.path.exists(os.path.join(persistent_folder, encoder_file)):
        print('Downloading encoder weights...')
        download_blob(bucket_name, encoder_file, os.path.join(persistent_folder, encoder_file))

    if not os.path.exists(os.path.join(persistent_folder, decoder_file)):
        print('Downloading decoder weights...')
        download_blob(bucket_name, decoder_file, os.path.join(persistent_folder, decoder_file))

    if not os.path.exists(os.path.join(image_path, test_image_file)):
        print('Downloading test image...')
        download_blob(bucket_name, test_image_file, os.path.join(image_path, test_image_file))


    print('Done!')

if __name__ == "__main__":
    download_transformer_model()