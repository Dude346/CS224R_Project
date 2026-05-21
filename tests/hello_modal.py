import modal

app = modal.App("hello-modal")

image = modal.Image.debian_slim()


@app.function(image=image)
def hello():
    print("hello from Modal")
    try:
        import torch

        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    except ImportError:
        print("no torch")


@app.local_entrypoint()
def main():
    hello.remote()
