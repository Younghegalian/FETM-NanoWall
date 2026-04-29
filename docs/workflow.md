# SEM Nanowall Domain Workflow

## Scientific Assumptions

- The SEM image contains graphene nanowalls on an approximately flat substrate.
- DA3 supplies a relative morphology field, not a trusted absolute SEM topography.
- The known height scale, currently `1.7 um`, is imposed after bias correction.
- Low-value regions in the oriented height field are treated as exposed substrate and
  are used to fit a planar bias.

## Recommended Config Procedure

1. Measure or mark the ruler.
   - Best: set `calibration.pixel_size_um` directly.
   - Good: set `ruler_line_px` as `[x1, y1, x2, y2]` and `ruler_length_um`.
   - Fast: set `ruler_bbox_px` around the scale bar and `ruler_length_um`.
2. Set `crop_px` to include only the graphene nanowall domain.
3. Run once and inspect `height_preview.png`.
4. If walls are inverted, toggle `depth.invert_depth`.
5. Tune `bias.base_percentile` if base regions are sparse or if nanowalls leak into the
   plane fit.
6. Keep `height.scale_percentile` below 100 to avoid one hot pixel setting the full
   physical height scale.

## Output Contract

`domain.npz` is the handoff artifact for later simulation code:

- `height_um`: 2D float32 surface height field.
- `pixel_size_um_x`, `pixel_size_um_y`: physical spacing of the height grid.
- `base_mask`: pixels used as substrate candidates for plane correction.
- `bias_plane`: fitted plane removed from the oriented depth map.
- `raw_depth`: original DA3 or precomputed depth map.
- `metadata_json`: JSON string with config and provenance.
