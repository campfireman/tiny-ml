import argparse
from datetime import datetime

from jobs import Deployment, Evaluation, FeatureExtraction, Optimization, Training
from utils import find_start, random_word_pair

PIPELINE = [
    FeatureExtraction,
    Training,
    Evaluation,
    Optimization,
    Deployment,
]


def main():
    parser = argparse.ArgumentParser("Training Pipeline")
    parser.add_argument("--job_name", type=str, required=False)
    parser.add_argument("--run_id", type=str, required=False)
    args = parser.parse_args()

    job_name = args.job_name.lower(
    ) if args.job_name else PIPELINE[0].get_job_name()
    run_id = args.run_id

    if job_name != PIPELINE[0].get_job_name() and not run_id:
        raise ValueError(
            f"When specifying job {job_name} run_id must be given!")

    if not run_id:
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random_word_pair()}"

    start = find_start(job_name, PIPELINE)

    if start == None:
        raise ValueError(f"No pipeline job with name \"{job_name}\" found!")

    input = None
    for job in PIPELINE[start:]:
        job = job(run_id)
        print(f"[ ] Running job {job}...")
        input = job.execute(input)
        print(f"[x] Job {job} done!")


if __name__ == "__main__":
    main()
