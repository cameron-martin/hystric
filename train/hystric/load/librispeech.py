from re import split
from typing import Generator, Iterable, List, Set, Tuple, TypeVar
import tensorflow as tf
import os
from pathlib import Path
import urllib
import tarfile
import itertools
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from .common import get_data_dir

CHUNK_SIZE = 500
THREAD_COUNT = 12

T = TypeVar('T')

def chunk(iterable: Iterable[T], n: int) -> Generator[List[T], None, None]:
    it = iter(iterable)
    while True:
        chunk_list = list(itertools.islice(it, n))
        if len(chunk_list) == 0:
            return
        yield chunk_list

_urls = [
    'https://www.openslr.org/resources/12/dev-clean.tar.gz',
    'https://www.openslr.org/resources/12/dev-other.tar.gz',
    'https://www.openslr.org/resources/12/test-clean.tar.gz',
    'https://www.openslr.org/resources/12/test-other.tar.gz',
    'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'https://www.openslr.org/resources/12/train-clean-360.tar.gz',
    'https://www.openslr.org/resources/12/train-other-500.tar.gz'
]

def download_files(data_dir: Path):
    downloads_dir = data_dir / 'downloads'
    downloads_dir.mkdir(exist_ok=True, parents=True)
    downloads: List[Path] = []
    for download_url in _urls:
        filename = os.path.basename(download_url)
        download_target = downloads_dir / filename
        downloads.append(download_target)
        if not download_target.exists():
            print(f"Downloading {download_url}... ", end='', flush=True)
            incomplete_target = downloads_dir / (filename + '.incomplete')
            incomplete_target.unlink(missing_ok=True)
            urllib.request.urlretrieve(download_url, filename=incomplete_target)
            incomplete_target.rename(download_target)
            print("Done")
    return downloads

def extract_files(downloads: List[Path], data_dir: Path):
    extracted_dir = data_dir / 'extracted'
    extracted_dir.mkdir(exist_ok=True, parents=True)
    for download in downloads:
        extraction_test = extracted_dir / 'LibriSpeech' / download.stem.split('.')[0]
        if not extraction_test.exists():
            print(f"Extracting {download}... ", end='', flush=True)
            file = tarfile.open(download, 'r')
            file.extractall(extracted_dir)
            print("Done")

def get_splits(data_dir: Path):
    extracted_dir = data_dir / 'extracted' / 'LibriSpeech'
    return [split_dir.name for split_dir in extracted_dir.iterdir() if split_dir.is_dir()]

def get_examples(split: str, data_dir: Path):
    split_dir = data_dir / 'extracted' / 'LibriSpeech' / split
    for dir1 in split_dir.iterdir():
        for dir2 in dir1.iterdir():
            transfile = dir2 / (f'{dir1.name}-{dir2.name}.trans.txt')
            lines = transfile.read_text().splitlines(keepends=False)
            for line in lines:
                id, label = line.split(" ", maxsplit=1)
                yield (dir2 / f"{id}.flac"), label

def decode_audio(file: Path):
    return AudioSegment.from_file(file, "flac").raw_data

def create_example(audio_file: Path, label: str) -> tf.train.Example:
    feature = {
        'audio': tf.train.Feature(bytes_list=tf.train.BytesList(value=[decode_audio(audio_file)])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_chunk(split: str, chunk: List[Tuple[Path, str]], i: int, data_dir: Path):
    print(f"Converting chunk {i} of {split}")
    directory = data_dir / 'tfrecords' / split
    directory.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(directory / f"{i}.tfrecord")) as writer:
        for audio_file, label in chunk:
            example = create_example(audio_file, label)
            writer.write(example.SerializeToString())
    print(f"Finished converting chunk {i} of {split}")

def check_futures(futures: Set[concurrent.futures.Future]):
    for future in futures:
        exception = future.exception()
        if exception is not None:
            raise exception
        

def convert_split(split: str, data_dir: Path):
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executer:
        futures = set()
        for i, my_chunk in enumerate(chunk(get_examples(split, data_dir), CHUNK_SIZE)):
            futures.add(executer.submit(convert_chunk, split, my_chunk, i, data_dir))
            if len(futures) >= THREAD_COUNT:
                print("Waiting for chunks to finish")
                done_and_not_done_futures = concurrent.futures.wait(futures, return_when='FIRST_COMPLETED')
                check_futures(done_and_not_done_futures.done)
                futures = done_and_not_done_futures.not_done
        check_futures(concurrent.futures.wait(futures).done)

# Create a description of the features.
feature_description = {
    'audio': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def parse_example(example_proto):
  example = tf.io.parse_single_example(example_proto, feature_description)
  audio = tf.io.decode_raw(example['audio'], out_type='int16')
  return audio, example['label']

def load_dataset(split_name: str, data_dir: Path):
    files = [str(dir) for dir in (data_dir / 'tfrecords' / split_name).iterdir()]
    return tf.data.Dataset.from_tensor_slices(files).map(lambda file: tf.data.TFRecordDataset(file).map(parse_example))

def load_librispeech(splits: List[str]) -> Tuple[tf.data.TFRecordDataset, ...]:
    data_dir = get_data_dir() / 'librispeech_custom'
    downloads = download_files(data_dir)
    extract_files(downloads, data_dir)
    for split in get_splits(data_dir):
        if not (data_dir / 'tfrecords' / split).exists():
            convert_split(split, data_dir)
    return tuple(map(lambda split_name: load_dataset(split_name, data_dir, shuffle), splits))