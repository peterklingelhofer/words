from enum import Enum
from typing import List, Tuple, Union
import random
import nltk
from nltk.corpus import words
from nltk.tag import pos_tag
from pyphen import Pyphen

nltk.download("averaged_perceptron_tagger")


class WordType(Enum):
    ADJECTIVE = "JJ"
    NOUN = "NN"
    ADVERB = "RB"
    VERB = "VB"
    CONJUNCTION = "CC"
    DETERMINER = "DT"
    PREPOSITION = "IN"
    PRONOUN = "PRP"
    INTERJECTION = "UH"
    CARDINAL = "CD"
    EXISTENTIAL = "EX"
    FOREIGN_WORD = "FW"
    LIST_MARKER = "LS"
    MODAL = "MD"
    NON_THIRD_PERSON_SINGULAR_PRESENT = "VBP"
    PAST_PARTICIPLE = "VBN"
    PAST_TENSE = "VBD"
    THIRD_PERSON_SINGULAR_PRESENT = "VBZ"
    WH_PRONOUN = "WP"
    POSSESSIVE_WH_PRONOUN = "WP$"
    WH_ADVERB = "WRB"


def get_syllable_count(word: str, lang: str = "en_US") -> int:
    dic: Pyphen = Pyphen(lang=lang)
    return len(dic.inserted(word).split("-"))


def generate_words(
        word_types: Union[WordType, List[WordType], None] = None,
        min_syllables: int = 1,
        max_syllables: int = 5,
        num_words: int = 10
) -> List[str]:
    if isinstance(word_types, WordType):
        word_types = [word_types]

    word_type_values = [wt.value for wt in word_types]

    nltk.data.find("corpora/words")
    word_list: List[str] = words.words()

    tagged_words: List[Tuple[str, str]] = pos_tag(word_list)

    filtered_words: List[str] = [
        word for word, pos in tagged_words if
        pos in word_type_values and min_syllables <= get_syllable_count(word) <= max_syllables
    ]

    random.shuffle(filtered_words)
    return filtered_words[:num_words]


if __name__ == "__main__":
    words_to_generate: List[str] = generate_words(
        word_types=[
            WordType.ADJECTIVE,
            WordType.NOUN,
            WordType.ADVERB,
            WordType.VERB,
        ],
        min_syllables=2,
        max_syllables=5,
        num_words=200,
    )

    for generated_word in words_to_generate:
        print(generated_word)
