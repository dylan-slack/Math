"""Main"""
import argparse

import transformers




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's do math.")
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=10)

