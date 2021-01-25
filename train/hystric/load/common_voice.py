from typing import Generator, Iterable, List, Set, Tuple, TypeVar
import tensorflow as tf
from pathlib import Path
import tarfile
import itertools
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import csv
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

def ensure_downloaded(data_dir: Path):
    downloads_dir = data_dir / 'downloads'
    download_file = downloads_dir / 'en.tar'
    if not download_file.exists():
        raise f"You must download the librispeech dataset to {download_file}"
    return download_file

def extract_file(download_file: Path, data_dir: Path):
    extracted_dir = data_dir / 'extracted'
    if not extracted_dir.exists():
        extracted_dir.mkdir(parents=True)
        print(f"Extracting {download_file}... ", end='', flush=True)
        file = tarfile.open(download_file, 'r')
        file.extractall(extracted_dir)
        print("Done")

def get_splits(data_dir: Path):
    extracted_dir = data_dir / 'extracted' / 'LibriSpeech'
    return [split_dir.name for split_dir in extracted_dir.iterdir() if split_dir.is_dir()]

def get_examples(split: str, data_dir: Path):
    extraction_dir = next((data_dir / 'extracted').iterdir())
    split_file = extraction_dir / 'en' / f'{split}.tsv'
    with open(split_file, newline='') as f:
        rd = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in rd:
            yield (extraction_dir / 'en' / 'clips' / row['path']), row['sentence']

def decode_audio(file: Path):
    return AudioSegment.from_file(file, "mp3").raw_data

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
    return tf.data.TFRecordDataset(files).map(parse_example)

def load_common_voice(splits: List[str]) -> Tuple[tf.data.TFRecordDataset, ...]:
    data_dir = get_data_dir() / 'common_voice_custom'
    download_file = ensure_downloaded(data_dir)
    extract_file(download_file, data_dir)
    for split in splits:
        if not (data_dir / 'tfrecords' / split).exists():
            convert_split(split, data_dir)
    return tuple(map(lambda split_name: load_dataset(split_name, data_dir), splits))