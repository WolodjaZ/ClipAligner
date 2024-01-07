# ClipAligner: Standing on the Shoulders of Giants to See and Read Further

PyTorch implementation and pretrained models for maybe future paper *ClipAligner: Standing on the Shoulders of Giants to See and Read Further*.

**ClipAligner** uses pretrained foundational models in image and text domains to align them in a common space. The alignment is done by training a alignment layers on top of foundational models on the aligned data image/captions pairs. The models are trained similar to the [CLIP](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Pretrained models

TODO

## Installation

TODO

## Usage

### Prepare data

TODO

### Add foundationel models

#### Text

TODO

#### Image

TODO

### Train

TODO

### Evaluate

TODO

## Notebooks

TODO

## Contributing

TODO

## License

State the type of license the project is under (link to the LICENSE file).

## Acknowledgments

TODO

## References

TODO

## Yet TODO

- Add multi optimizers and locking of layers like proposed in [OpenCLIP](https://github.com/mlfoundations/open_clip/blob/695c72d1ef1c3044404bb5579ea25e2051658c2d/src/open_clip/transformer.py#L435).
- Add CC12M Dataset with this [link](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)
- Add YFCC100M Dataset with this [link](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md)
- Add accumulations of embeddings.
