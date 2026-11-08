# Publishing the Community Documentation

This guide lists the steps to publish the community documentation on Hugging Face: the article, the model card, the results dataset and a collection that groups them. Nothing here has been published yet. Every step needs a Hugging Face write token, which should be revoked once the last step is done.

Start only after the live Space has been accepted.

### 1. Model card

The card template is `deploy/model/README.md`. `make deploy` fills in `{revision}` and `{space}`, uploads it, uploads the model package and then pins the application to the last commit that changed `manifest.json`. A changed card or image is a new commit on the model repository, but it does not change the manifest, so the application keeps its revision and readers keep their cached models.

The card shows the conversion pipeline as a Mermaid block followed by an image, `images/conversion-pipeline.svg`, since model cards show Mermaid blocks as code (see step 3).

1. Upload the image to the model repository:

   ```python
   from huggingface_hub import HfApi

   HfApi().upload_file(
       path_or_fileobj="docs/community/diagrams/conversion-pipeline.svg",
       path_in_repo="images/conversion-pipeline.svg",
       repo_id="sinsankio/sinhalallm-1.7b-onnx-web",
       commit_message="Add the conversion pipeline diagram",
   )
   ```

2. Merge `docs/community`, tag the release and run `make deploy`.
3. Check the card at [sinsankio/sinhalallm-1.7b-onnx-web](https://huggingface.co/sinsankio/sinhalallm-1.7b-onnx-web), including the image, and check that the printed model revision is the same as in the previous release.

### 2. Results dataset

1. Build the tables into the model store (`<store>/dataset/`). `make` reads the store location from `local.mk`:

   ```sh
   make dataset
   ```

2. Review `perplexity.csv`, `agreement.csv`, `per_sentence.csv`, `browser.csv` and `README.md`, and confirm the license in the dataset card (CC BY-SA 4.0, since the scores are derived from FLORES+).
3. Create the dataset and upload the folder:

   ```python
   from huggingface_hub import HfApi

   api = HfApi()
   repo = "sinsankio/sinhalallm-1.7b-onnx-web-results"
   api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
   api.upload_folder(
       folder_path="<store>/dataset",
       repo_id=repo,
       repo_type="dataset",
       commit_message="Add evaluation results",
   )
   ```

4. Open the dataset page and check that the data viewer shows the four tables.

### 3. Article

The article is `docs/community/article.md`. The diagrams are in `docs/community/diagrams/`, as Mermaid sources (`.mmd`) and rendered images (`.svg`). Each diagram appears in the article as a Mermaid block followed by its image.

Hugging Face shows Mermaid blocks in model and dataset cards as plain code: on 2026-11-08, the cards of `Houzeric/text-to-mermaid`, `datasets/RonnieAllen/mermaid-flowchart-dataset` and `datasets/Kartick/problem-to-cfg-mermaid` each contained a well-formed Mermaid block, and after the page had run its scripts in Chrome, each showed it as a code block with no diagram. No community article with a Mermaid block was found, so the article editor is unconfirmed. Check the preview before publishing: if it draws the Mermaid blocks, remove the images after them; if it shows them as code, keep both, or remove the blocks and keep the images.

After editing a diagram, update its block in the article and render its SVG with plain text labels, since HTML labels can disappear when an SVG is shown as an image:

```sh
echo '{"htmlLabels":false,"flowchart":{"htmlLabels":false}}' > mermaid.json
npx -y @mermaid-js/mermaid-cli@12.0.0 -c mermaid.json -b white -i docs/community/diagrams/<name>.mmd -o docs/community/diagrams/<name>.svg
```

Community articles on huggingface.co need a PRO subscription or a write role in a Team or Enterprise organization, which this account does not have. The article is therefore published as the static Space [sinsankio/sinhalalm-offline-article](https://huggingface.co/spaces/sinsankio/sinhalalm-offline-article), which renders `article.md` in the browser and draws the Mermaid blocks itself, hiding the image after each block.

1. Check that the dataset link in the article opens the dataset published in step 2.
2. Publish or update the Space:

   ```sh
   make article
   ```

3. Open the Space in light and dark mode and check that the six diagrams are drawn and the Sinhala text renders.
4. To move it to a community article later, paste `article.md` at huggingface.co/new-blog and upload the six SVGs where the article shows `diagrams/<name>.svg`.

### 4. Collection

Collections hold models, datasets, Spaces and papers. The article is a Space, so it is an item like the others.

```python
from huggingface_hub import HfApi

api = HfApi()
collection = api.create_collection(
    title="Offline Sinhala question answering in the browser",
    namespace="sinsankio",
    description="SinhalaLLM-1.7B running fully offline in the browser: the application, the article, the model files and the results.",
)
items = [
    ("sinsankio/sinhalalm-offline", "space", "The application"),
    ("sinsankio/sinhalalm-offline-article", "space", "The article: how it was built"),
    ("sinsankio/sinhalallm-1.7b-onnx-web", "model", "Browser-ready 8-bit and 4-bit ONNX files"),
    ("sinsankio/sinhalallm-1.7b-onnx-web-results", "dataset", "Quality, speed and memory results"),
    ("namindu94/SinhalaLLM-1.7B", "model", "The original model by Namindu Sanchila Nadeeshan"),
]
for item_id, item_type, note in items:
    api.add_collection_item(collection.slug, item_id=item_id, item_type=item_type, note=note)
print(collection.url)
```

### 5. Links and clean-up

1. Add the article and collection links to `deploy/space/README.md` and `README.md`, and release the Space README with the next `make deploy`.
2. Revoke the write token at huggingface.co/settings/tokens, and delete `~/.cache/huggingface/token` or replace it with a read-only token.
