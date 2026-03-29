# Models for `cellm`

This directory is for local model artifacts used during development and testing.

## Publish with Git LFS

Track these two checkpoints with Git LFS:

- `models/smollm2-135m-int8.cellm`
- `models/smolvlm-256m-int8.cellm`

Example:

```bash
git lfs track "models/smollm2-135m-int8.cellm"
git lfs track "models/smolvlm-256m-int8.cellm"
git add .gitattributes models/smollm2-135m-int8.cellm models/smolvlm-256m-int8.cellm
```

## Keep lightweight files in git

- docs/manifests/checksums
- sample images in `models/test_images/`
