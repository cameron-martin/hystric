from pathlib import Path
from .common import get_data_dir
from typing import Dict
import urllib.request

def download_if_not_exists(url: str, target: Path):
    if not target.exists():
        urllib.request.urlretrieve(url, filename=target)

def download_cmu(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    download_if_not_exists('http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b', data_dir / 'cmu-0.7b')
    download_if_not_exists('http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols', data_dir / 'cmu-0.7b.symbols')

def create_phoneme_mapping(data_dir: Path):
    symbols_path = data_dir / 'cmu-0.7b.symbols'
    mapping = {}
    current_index = 1
    with symbols_path.open() as f:
        for line in f:
            phoneme_symbol = line.strip()
            if phoneme_symbol == '': continue
            base_phoneme_symbol = phoneme_symbol.rstrip('1234567890')
            if base_phoneme_symbol in mapping:
                mapping[phoneme_symbol] = mapping[base_phoneme_symbol]
            else:
                mapping[phoneme_symbol] = current_index
                current_index += 1
    return mapping, (current_index - 1)
            
def create_dictionary(phoneme_mapping: Dict[str, int], data_dir: Path):
    dictionary_path = data_dir / 'cmu-0.7b'
    dictionary = {}
    with dictionary_path.open(encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;'): continue
            word, *phonemes = line.split(' ')
            if word.endswith(')'): continue # Skip alternate translations
            phonemes = (phoneme.strip() for phoneme in phonemes)
            dictionary[word] = [phoneme_mapping[phoneme] for phoneme in phonemes if phoneme != '']
    return dictionary

def load_cmu():
    data_dir = get_data_dir() / 'cmu_dictionary'
    download_cmu(data_dir)
    phoneme_mapping, alphabet_size = create_phoneme_mapping(data_dir)
    dictionary = create_dictionary(phoneme_mapping, data_dir)
    
    return dictionary, phoneme_mapping, alphabet_size