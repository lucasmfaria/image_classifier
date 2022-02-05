from pathlib import Path
import imghdr
import tensorflow as tf
import os
from tqdm import tqdm


DEFAULT_DATASET_SOURCE_PATH = Path(__file__).parent.parent / 'data' / 'dataset'


def main(dataset_source_path=DEFAULT_DATASET_SOURCE_PATH):
    directory_list = [directory for directory in Path(dataset_source_path).iterdir() if
                      os.path.isdir(directory)]
    files_with_problem = []
    for directory in directory_list:
        print("Reading class:", directory.resolve())
        total_files = len(list(directory.iterdir()))
        for file in tqdm(directory.iterdir(), total=total_files):
            if imghdr.what(file) is not None:
                try:
                    tensor = tf.io.read_file(str(file))
                    _ = tf.image.decode_image(tensor)
                except:
                    print("Problem with:", file)
                    files_with_problem.append(file)
    return files_with_problem


if __name__ == '__main__':
    files = main()
    print("Image files with problem:", files)
