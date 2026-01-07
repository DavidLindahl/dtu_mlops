import typer

def main(name: str = "World", count: int = 1):
    for _ in range(count):
        print(f"Hello {name}!")

if __name__ == "__main__":
    typer.run(main)
