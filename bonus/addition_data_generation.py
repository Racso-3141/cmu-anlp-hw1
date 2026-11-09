# Generate a dataset of addition problems
import random
import os

def generate_addition_data(n):
    """
    Generate a dataset of n unique 4-digit addition problems.

    The function should:
    1) Randomly sample two 4-digit integers a and b.
        Recall: 4-digit integers are integenrs in the range of [1000, 9999]
    2) Compute their sum c = a + b
    3) Ensure uniqueness of addition pairs, treating (a, b) and (b, a) as identical
    4) Repeat until n unique examples are collected
    5) Return the data formatted as strings of the form "a+b=c"
    """
    # There are only this many unordered pairs (a, b) with a, b in [1000, 9999]:
    # C(9000, 2) + 9000, counting a == b. Asking for more can never be satisfied,
    # and a naive rejection-sampling loop would spin forever instead of failing.
    MAX_UNIQUE = 9000 * 8999 // 2 + 9000
    if n > MAX_UNIQUE:
        raise ValueError(
            f"asked for {n:,} unique problems but only {MAX_UNIQUE:,} exist. "
            f"This usually means your model is so small that the capacity budget "
            f"implies more data than the task can provide — see --capacity."
        )

    # todo: sample unique unordered 4-digit pairs, format as "a+b=c"
    raise NotImplementedError

def generate_dataset(n, filename, save_dir="data"):
    data = generate_addition_data(n)
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        f.write('\n'.join(data))
    print(f"{n} data points saved to {filepath}")
