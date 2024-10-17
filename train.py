
from model import GPTModel, GPTConfig


def main():
    config = GPTConfig()
    print(config)

    model = GPTModel(config)
    optimizer = model.configure_optimizers(learning_rate=5e-4)


if __name__ == "__main__":
    main()
