import os
import subprocess
import time
import shutil

S3_BUCKET_NAME = 'boda-ts-bucket'
S3_INPUT_PREFIX = 'input'
S3_OUTPUT_PREFIX = 'tmp-ts'
LOCAL_TS_PATH = os.path.expanduser('~/codes/ts-s3-input')
PROCESS_TS_PATH = os.path.expanduser('~/codes/ts-process')
PROCESSED_TS_PATH = os.path.expanduser('~/codes/ts-processed')

def sync_s3_to_local():
    sync_command = f"aws s3 sync s3://{S3_BUCKET_NAME}/{S3_INPUT_PREFIX} {LOCAL_TS_PATH} --delete"
    subprocess.run(sync_command, shell=True, check=True)

def sync_processed_to_s3():
    sync_command = f"aws s3 sync {PROCESSED_TS_PATH} s3://{S3_BUCKET_NAME}/{S3_OUTPUT_PREFIX} --delete"
    subprocess.run(sync_command, shell=True, check=True)

def copy_new_files(source_path, dest_path, processed_files):
    source_files = os.listdir(source_path)
    for file_name in source_files:
        if file_name not in processed_files and file_name.endswith('.ts'):
            shutil.copy(os.path.join(source_path, file_name), os.path.join(dest_path, file_name))
            processed_files.add(file_name)
            print(f"Copied: {file_name}")

def main():
    processed_files = set()
    while True:
        sync_s3_to_local()
        copy_new_files(LOCAL_TS_PATH, PROCESS_TS_PATH, processed_files)
        sync_processed_to_s3()
        time.sleep(30)

if __name__ == "__main__":
    main()
