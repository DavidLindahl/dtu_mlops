import typer


def main(name: str = "World", count: int = 1):
    """Print a greeting multiple times.

    Args:
        name: The name to greet.
        count: The number of times to greet.
    """
    for _ in range(count):
        print(f"Hello {name}!")

if __name__ == "__main__":
    typer.run(main)
