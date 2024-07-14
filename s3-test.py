import boto3
import os

def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File downloaded successfully from {bucket_name}/{s3_file_key} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def upload_file_to_s3(bucket_name, local_file_path, s3_file_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_key)
        print(f"File uploaded successfully from {local_file_path} to {bucket_name}/{s3_file_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    bucket_name = "boda-ts-bucket"
    s3_file_key = "v4_nano_results.pt"
    local_file_path = "./v4_nano_results.pt"
    output_s3_file_key = "output.ts"

    # Step 1: Download file from S3
    download_file_from_s3(bucket_name, s3_file_key, local_file_path)

    # Step 2: Upload file to S3 as output.ts
    # upload_file_to_s3(bucket_name, local_file_path, output_s3_file_key)
