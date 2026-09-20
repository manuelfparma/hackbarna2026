import re

from agent.nodes.catalog import clean_title

ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
NUMBERS = ("one", "two", "three", "four", "five", "six", "seven", "eight")


def select_movie(answer, movies):
    text = answer.strip().casefold().rstrip(".!")
    if text.isdigit():
        index = int(text) - 1
        return movies[index] if 0 <= index < len(movies) else None
    for index, (ordinal, number) in enumerate(zip(ORDINALS, NUMBERS)):
        if re.fullmatch(
            rf"(?:the )?{ordinal}(?: one| movie)?|number (?:{number}|{index + 1})", text
        ):
            return movies[index] if index < len(movies) else None
    matches = [
        movie
        for movie in movies
        if clean_title(movie).casefold() == clean_title(text).casefold()
    ]
    return matches[0] if len(matches) == 1 else None


def match_title(title, movies):
    movies = list(dict.fromkeys(movies))
    text = clean_title(title).casefold()
    exact = [movie for movie in movies if clean_title(movie).casefold() == text]
    if len(exact) == 1:
        return exact[0]
    partial = [
        movie for movie in movies if text and text in clean_title(movie).casefold()
    ]
    return partial[0] if len(partial) == 1 else None
