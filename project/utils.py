import random

ADJECTIVES = [
    "cheerful", "playful", "brilliant", "friendly", "eager", "happy",
    "jolly", "gentle", "kind", "lively", "witty", "bubbly", "sunny", "sparkling"
]
NOUNS = [
    "ada", "turing", "hopper", "compiler", "lambda", "socket", "protocol",
    "array", "kernel", "stack", "recursion", "segment", "pointer", "scheduler",
    "quine", "buffer", "thread", "bytecode", "pipeline", "neuron"
]


def random_word_pair():
    return f"{random.choice(ADJECTIVES)}_{random.choice(NOUNS)}"


def find_start(job_name, pipeline):
    for i, job in enumerate(pipeline):
        if job.get_job_name() == job_name:
            return i
    return None
