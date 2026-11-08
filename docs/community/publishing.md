# Publishing the Community Documentation

This guide lists the steps to publish the community documentation on Hugging Face: the article, the model card, the results dataset and a collection that groups them. Nothing here has been published yet. Every step needs a Hugging Face write token, which should be revoked once the last step is done.

Start only after the live Space has been accepted.

### 1. Model card

The card template is `deploy/model/README.md`. `make deploy` fills in `{revision}` and `{space}`, uploads it, uploads the model package and then pins the application to the latest commit of the model repository.

A changed card is a new commit on the model repository, even when the model files stay the same. With the current `deploy/deploy_space.py`, the next release would therefore pin the application to that new commit, and every reader's browser would delete its cached models and download them again (1.2 to 2.5 GB). Before the card is published, `deploy/deploy_space.py` must pin the application to the last commit that changed the model files, not to the latest commit.

1. Fix the pinning in `deploy/deploy_space.py` as described above, and test it with a card-only change.
2. Merge `docs/community`, tag the release and run `make deploy`.
3. Check the card at [sinsankio/sinhalallm-1.7b-onnx-web](https://huggingface.co/sinsankio/sinhalallm-1.7b-onnx-web), and check that the application is still pinned to the same model revision as before.

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

The article is `docs/community/article.md`. The diagrams are in `docs/community/diagrams/`, as Mermaid sources (`.mmd`) and rendered images (`.svg`). The Hugging Face documentation does not mention Mermaid support for articles or cards, so the article uses the images. After editing a diagram, render its SVG with plain text labels, since HTML labels can disappear when an SVG is shown as an image:

```sh
echo '{"htmlLabels":false,"flowchart":{"htmlLabels":false}}' > mermaid.json
npx -y @mermaid-js/mermaid-cli@12.0.0 -c mermaid.json -b white -i docs/community/diagrams/<name>.mmd -o docs/community/diagrams/<name>.svg
```

1. Replace the `[[NEEDS INPUT: …]]` placeholder for the dataset link with the link from step 2.
2. On huggingface.co, start a new community article from your profile, and paste the Markdown.
3. Upload the six images in the editor and replace each `diagrams/<name>.svg` path with the uploaded image. The SVG labels are plain SVG text, so the text in an SVG depends on the reader's installed fonts. For `answer-flow`, which contains Sinhala, a PNG is the safer choice, and PNG versions of any diagram are rendered with:

   ```sh
   npx -y @mermaid-js/mermaid-cli@12.0.0 -b white -s 2 -i docs/community/diagrams/<name>.mmd -o <name>.png
   ```

4. Preview the article in light and dark mode, check that the Sinhala text renders, and publish it.

### 4. Collection

Collections hold models, datasets, Spaces and papers, but not articles, so the article link goes into the description.

```python
from huggingface_hub import HfApi

api = HfApi()
collection = api.create_collection(
    title="Offline Sinhala question answering in the browser",
    namespace="sinsankio",
    description="SinhalaLLM-1.7B running fully offline in the browser. Article: <article link>",
)
items = [
    ("sinsankio/sinhalalm-offline", "space", "The application"),
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
