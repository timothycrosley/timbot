import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional
import httpx
from tqdm.asyncio import tqdm


class AsyncFileDownloader:
    def __init__(self, max_concurrent: int = 5, chunk_size: int = 8192):
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def download_file(
        self,
        client: httpx.AsyncClient,
        url: str,
        filepath: Path,
        progress_bar: Optional[tqdm] = None,
    ) -> bool:
        """Download a single file with progress tracking"""
        async with self.semaphore:  # Limit concurrent downloads
            try:
                async with client.stream(
                    "GET", url, follow_redirects=True
                ) as response:  # Added follow_redirects=True
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))

                    # Create progress bar for this file if not provided
                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc=filepath.name,
                        )

                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    async with aiofiles.open(filepath, "wb") as f:
                        async for chunk in response.aiter_bytes(self.chunk_size):
                            await f.write(chunk)
                            if progress_bar:
                                progress_bar.update(len(chunk))

                    if progress_bar:
                        progress_bar.close()

                    return True

            except Exception as e:
                print(f"Failed to download {url}: {e}")
                return False

    async def download_multiple(
        self,
        downloads: List[Dict[str, str]],  # [{"url": "...", "filepath": "..."}]
    ) -> Dict[str, bool]:
        """Download multiple files concurrently"""

        # Configure client with redirect following and longer timeout
        async with httpx.AsyncClient(
            timeout=60.0,  # Longer timeout for large files
            follow_redirects=True,  # Follow redirects automatically
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        ) as client:
            tasks = []

            for item in downloads:
                url = item["url"]
                filepath = Path(item["filepath"])

                # Skip if file already exists
                if filepath.exists():
                    print(f"Skipping {filepath.name} (already exists)")
                    continue

                task = self.download_file(client, url, filepath)
                tasks.append((task, url))

            if not tasks:
                print("No files to download")
                return {}

            print(f"Downloading {len(tasks)} files...")

            # Execute all downloads concurrently
            results = await asyncio.gather(
                *[task for task, _ in tasks], return_exceptions=True
            )

            # Map results back to URLs
            return {
                url: result if not isinstance(result, Exception) else False
                for (_, url), result in zip(tasks, results)
            }
