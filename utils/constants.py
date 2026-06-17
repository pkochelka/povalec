# EU&I serves identical statements for country locales that share a language, so
# we keep one code per language: at, be -> de; cy -> gr; lu -> fr are dropped.
LANGS = [
    "bg", "hr", "cz", "dk", "ee", "fi", "fr",
    "de", "gr", "hu", "ie", "it", "lv", "lt", "mt", "nl",
    "pl", "pt", "ro", "sk", "si", "es", "se",
]

ALL_LANGS = LANGS + ["en"]
ALL_LANGS_STR = ",".join(ALL_LANGS)