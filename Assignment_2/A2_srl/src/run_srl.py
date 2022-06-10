import argparse

from assignment2_srl import run_srl

def main_func(args):
    run_srl(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRL: CoNLL 2009")

    # data
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="the input data directory")
    parser.add_argument("--print_stats", type=str, default=False,
                        help="whether print the statistics for all features")

    args = parser.parse_args()
    print(args)
    main_func(args)
