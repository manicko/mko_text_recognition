from pickle import FALSE

from hunspell import Hunspell


LANG_MAP = {
    "uzb_cyrl": "uz",  # Узбекский (кириллица)
    "bs": "bs_BA",  # Боснийский
    "srp_cyrl": "sr",  # Сербский (кириллица)
    "srp": "sr-Latn",  # Сербский (латиница, если доступен)
    "srp_latn": "sr-Latn",  # Сербский (латиница, если доступен)
    "rus": "ru_RU"
}

DICT_PATH = f'C:\hunsdicts'

def spellcheck_with_hunspell(text: str, lang_from: str, keep_misspelled = False) -> str:
    """
    Проверяет орфографию текста с использованием Hunspell.

    :param text: Строка текста для анализа.
    :param lang_from: Язык текста, заданный в формате Tesseract (e.g., "uzb_cyrl").
    :param keep_misspelled: Если True, будет рядом с некорректным словом предлагать замену в скобках []
    :return: Скорректированный текст
    """
    # Определяем язык Hunspell
    hunspell_lang = LANG_MAP.get(lang_from)
    if not hunspell_lang:
        raise ValueError(f"Unsupported language: {lang_from}")

    try:
        # Инициализация Hunspell со словарём
        h = Hunspell(hunspell_lang, hunspell_data_dir=DICT_PATH)

    except Exception as e:
        raise RuntimeError(f"Error initializing Hunspell for language '{hunspell_lang}': {e}")

    # Разделяем текст на слова (основной разделитель — пробел)
    words = text.split()

    correct_words = []
    misspelled_words = []
    # Проверяем каждое слово
    for word in words:
        if h.spell(word):
            correct_words.append(word)
        else:
            if keep_misspelled:
                correct_words.append(word)
            if suggestions := h.suggest(word):
                best_word = f'[{suggestions[0]}]' if keep_misspelled else suggestions[0]
                correct_words.append(best_word)

    return ' '.join(correct_words)
