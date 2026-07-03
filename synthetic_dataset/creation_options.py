# In this file I define the lists of possible values that the parameters can assume

political_leanings = {"far_left": .01, "left": .01, "center": .005, "right": .2, "far_right": .52, "non_political": .215, "unknown": .04}

posts_levels = {0: .61, 1: .15, 2: .19, 3: .01, 4: .03, 5: .01}
#posts_levels = {0: .0, 1: .0, 2: .0, 3: .0, 4: .5, 5: .5}
interests = ["basketball", "football", "tennis", "american football", "rugby", "golf", "athletics", "reading books",
             "sports", "travel", "cook", "fitness", "gardening", "politics", "technology", "self improvement",
             "history", "science", "animals", "country music", "rap music", "classical music", "pop music",
             "video games", "podcasts", "horror movies", "action movies", "sci-fi movies", "writing", "ecology", "yoga"]

professions_under_25 = ["waiter", "student", "baby sitter", "fast-food worker"]
professions_over_25 = ["unemployed", "nurse", "vet", "driver", "janitor", "farmer", "mason", "plumber", "hairdresser",
                       "lawyer", "teacher", "accountant", "journalist", "software developer", "engineer"]

ethnicities = {"caucasian": .6, "asian": .07, "black": .13, "mexican/latino": .2}
religions = {"christian (evangelical protestant)": .24, "christian (catholic)": .2, "christian (black protestant)": .05,
             "christian (mormon)": .02, "christian (mainline protestant)": .11, "jewish": .02, "muslim": .01,
             "buddhist": .01, "hindu": .01, "atheist": .05, "agnostic": .06, "nothing in particular": .22}
christian_religions = {k: v for k, v in religions.items() if "christian" in k}
total = sum(christian_religions.values())
christian_religions_norm = {k: v/total for k, v in christian_religions.items()}


female_keywords = ["mom", "mum", "mother", "wife", "grandmother"]
male_keywords = ["dad", "father", "husband", "grandfather"]

us_states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

professions = professions_under_25 + professions_over_25