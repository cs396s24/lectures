import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def observed(n=100, c_dim=6, ols="y ~ a"):
    """
    The observed data distribution
      C: roll a k-sided die and record the result
      A: flip `1 + k - C` fair coins, and record 1 if at least one flip lands heads
      Y: flip `C + A` fair coins, and record the number of heads
    """

    c = np.random.randint(1, 1 + c_dim, n)
    a = np.random.binomial(n=1 + c_dim - c, p=0.5, size=n)
    a = (a > 0).astype(np.int32)
    y = np.random.binomial(n=a + c, p=0.5)
    df = pd.DataFrame(data=dict(c=c, a=a, y=y))
    return smf.ols(ols, data=df).fit().params['a']


def randomized(n=100, c_dim=6, ols="y ~ a"):
    """
    The same distribution, except A is replaced with a fair coin f
      C: roll a k-sided die and record the result
      A: flip a single fair coin, and record 1 if it lands heads
      Y: flip `C + A` fair coins, and record the number of heads
    """

    c = np.random.randint(1, 1 + c_dim, n)
    a = np.random.binomial(n=1, p=0.5, size=n)
    y = np.random.binomial(n=a + c, p=0.5)
    df = pd.DataFrame(data=dict(c=c, a=a, y=y))
    return smf.ols(ols, data=df).fit().params['a']


if __name__ == "__main__":
    # Run an experiment with the given args
    #   dist: either "observed" or "randomized" distribution
    #   n: the number of samples to draw from the distribution
    #   c_dim: possible values that C can take (number of sides of the die)
    #   ols: regression model; either "y ~ a" or "y ~ a + c"

    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=str, choices=["observed", "randomized"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--c_dim", type=int, default=6)
    parser.add_argument("--ols", type=str, default="y ~ a")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(42)
    func = observed if args.func == "observed" else randomized
    results = [func(n=args.n, c_dim=args.c_dim, ols=args.ols)
               for i in range(args.repeats)]
    err = ""
    if args.repeats > 1:
        err = f" ± {np.std(results):.3f}"
    print(f"{np.mean(results):.3f}{err}")
