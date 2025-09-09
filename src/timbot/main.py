import typer
import asyncio
from async_downloader import AsyncFileDownloader

app = typer.Typer()


async def _download_models_async():
    downloads = [
        {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "filepath": "models/en_US-lessac-medium.onnx",
        },
        {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            "filepath": "models/en_US-lessac-medium.onnx.json",
        },
    ]

    downloader = AsyncFileDownloader(max_concurrent=3)
    results = await downloader.download_multiple(downloads)

    for url, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {url}")
    return results


@app.command()
def download_models():
    """Installs models needed to run timbot."""
    asyncio.run(_download_models_async())


@app.command()
def start():
    """Starts timbot in interactive mode."""
    print("Starting will go here")


if __name__ == "__main__":
    app()
