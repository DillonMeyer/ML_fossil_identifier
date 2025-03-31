from duckduckgo_search import DDGS
import urllib.request
import os

def download_images(query, folder, max_images=50):
    os.makedirs(folder, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        for i, result in enumerate(results):
            url = result["image"]
            try:
                urllib.request.urlretrieve(url, f"{folder}/{i}.jpg")
            except Exception as e:
                print(f"Error downloading image {i}: {e}")

# Example usage
download_images("trilobite fossil", "fossils/trilobite", 100)
download_images("ammonite fossil", "fossils/ammonite", 100)
download_images("crinoid fossil", "fossils/crinoid", 100)