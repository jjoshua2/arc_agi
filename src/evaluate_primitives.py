import asyncio
import pickle
import os
import argparse


from devtools import debug

from src.data import eval_challenges, training_challenges, v2_training_challenges, v2_eval_challenges
from src.logic import can_library_solve_challenge
from src import logfire


async def main() -> None:
    # Add argument parser
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train_keys = list(training_challenges.keys())
    v2_train_keys = list(v2_training_challenges.keys())
    print(f"v2 training set size: {len(v2_train_keys)}")
    logfire.debug(f"v2 training set size: {len(v2_train_keys)}")
    #random.shuffle(v2_train_keys)
    train_ids_to_test = v2_train_keys
    debug(train_ids_to_test)
    logfire.debug(f"train_ids_to_test: {train_ids_to_test}")

    
    eval_ids_to_test = list(v2_eval_challenges.keys())
    print(f"v2 eval set size: {len(eval_ids_to_test)}")
    logfire.debug(f"v2 eval set size: {len(eval_ids_to_test)}")
    # take random 10 sample fro eval_ids_to_test
    debug(eval_ids_to_test)


    from src.models import Library

    # Function to load library
    def load_library(filename="saved_library_eval_90.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            print(f"No library file found at {filename}, creating new library")
            return Library(primitives=[])

    library_path = "saved_library_eval_100.pkl"
    library = load_library(library_path)

    print(f"library size: {len(library.primitives)}")
    logfire.debug(f"library size: {len(library.primitives)}")

    if len(library.primitives) == 0:
        print("No library found, exiting")
        return

    solved_challenges = []

    for challenge_id in eval_ids_to_test:
        debug(challenge_id)
        challenge = v2_eval_challenges[challenge_id]

        can_solve = await can_library_solve_challenge(
            challenge=challenge,
            library=library,
        )
        if can_solve:
            solved_challenges.append(challenge_id)
            print(f"Solved Challenge {challenge_id}")
        
    logfire.debug(f"Solved Challenges: {solved_challenges}")
    print(f"Solved Challenges: {solved_challenges}")
    logfire.debug(f"Correct Percent SO FAR: {len(solved_challenges) / len(eval_ids_to_test)}")
    print(f"Correct Percent SO FAR: {len(solved_challenges) / len(eval_ids_to_test)}")


if __name__ == "__main__":
    asyncio.run(main())
