import modal

app = modal.App("wandb-check")

image = modal.Image.debian_slim(python_version="3.11").pip_install("wandb")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=300,
)
def smoke_test() -> str:
    import os
    import random
    import wandb

    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY not found in env"

    run = wandb.init(
        project="cs224r-project",
        name="modal-smoke-test",
        config={"dummy": True},
    )
    for step in range(20):
        wandb.log(
            {
                "fake_reward": random.uniform(-1, 1) + step * 0.05,
                "fake_loss": 1.0 / (step + 1),
            },
            step=step,
        )
    url = run.url
    wandb.finish()
    return url


@app.local_entrypoint()
def main():
    url = smoke_test.remote()
    print(f"W&B run URL: {url}")
