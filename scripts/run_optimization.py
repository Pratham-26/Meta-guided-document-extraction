import argparse

from src.optimization.gepa import run_gepa_cycle


def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization cycle")
    parser.add_argument("--category", required=True, help="Category name")
    parser.add_argument(
        "--generations", type=int, help="Number of generations (overrides config)"
    )
    parser.add_argument(
        "--population", type=int, help="Population size (overrides config)"
    )
    args = parser.parse_args()

    print(f"Starting GEPA optimization for '{args.category}'...")
    result = run_gepa_cycle(
        category=args.category,
        generations=args.generations,
        population_size=args.population,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Optimization complete.")
    print(f"  Generations: {result['generations_run']}")
    print(f"  Final candidate: {result['final_candidate']}")
    print(f"  Fitness scores: {result['fitness_scores']}")


if __name__ == "__main__":
    main()
