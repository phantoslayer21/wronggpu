# Frame generation: formats and open items

What the clean side has for frame generation, and what it still waits for. **OPEN** marks behaviour that has not been
delivered by the spec side; the code never guesses it (it raises, or takes the choice as an explicit argument).

## Weights

- One weight set for every configuration: both FG presets (UI recomposition off and on), every multi-frame count and
  index, every combination of optional inputs. The same blobs are in every 310.x frame-generation library.
- Two networks, float16 elements, ReLU activations, no attention, no FP8:
  - **net1**: a six-level encoder-decoder on a 2 x 2-folded grid; 17 blobs, each `[weights][zero area][bias x 16]`.
  - **net2**: two small encoder-decoders (sub-nets A and B) of 3 x 3 convolutions; 25 layers, a weight blob and a bias
    blob each.
- Shapes and byte sizes per layer: [`layout.py`](layout.py) (checked by `tests/test_fg_layout.py`).

## Converter map

- Map version 1 with `"preset": "FG"` (converter README, "Preset FG"): one block source per blob, located by length
  and hashes, `f16` reads. net1's zero area is skipped by the tensor's base; its bias is read as copy 0 of each group of
  eight (`layout.net1_bias_index`).
- The spec side's map file is **OPEN** (not delivered). Until it arrives the converter is tested with our own random
  blobs and a stand-in map (`clean/converter/tests/cases.py`, `fg_map`), whose stage and tensor names and axis order are
  placeholders.
- A conversion of a real library will be checked by digests only (the converter compares every stage digest the map
  gives); no value or byte of it is ever written into the repository or any public text.

## OPEN (waiting for the spec side)

1. The converter map file.
2. The in-blob axis order of the weights (tap, input channel, output channel, any tiling), for both networks.
3. Which launch reads which blob (upload order is known; launch order is not).
4. net1's blob 2: a 1 x 1 convolution or a 2 x 2 one.
5. Both networks' grid sizes relative to the output size.
6. What the UI-recomposition preset changes in the stage order.
7. Launch order, the motion-field scatter, the pull pyramid and the UI handling (part 3).
8. Whether the networks run once per real frame or once per generated frame.
