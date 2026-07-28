# CricVision labeling guide

Read this **before** labeling a single frame, and re-read it when unsure.
Consistency beats volume: a smaller dataset labeled by one set of rules trains
better than a bigger one labeled by mood.

## Classes (7)

| id | class | definition |
|----|-------|------------|
| 0 | `batsman` | Either batter — striker or non-striker — while their side is batting. Includes a batter running, diving, or mid-runout: they stay `batsman` until the umpire's decision is visible. |
| 1 | `bowler` | The bowler from the moment the run-up starts until the follow-through ends. Outside that window (walking back to their mark, fielding off their own bowling after the follow-through) they are a `fielder`. |
| 2 | `wicketkeeper` | The keeper while in keeping position/gear behind the stumps. A keeper chasing a ball far from the stumps is still `wicketkeeper` (the gloves disambiguate). |
| 3 | `umpire` | On-field umpires only (main + square leg). Not the third umpire on a screen, not match referees. |
| 4 | `fielder` | Any other player of the bowling side. Substitutes count. |
| 5 | `ball` | The ball, whenever ≥ 3×3 px is visible. Label motion-blurred streaks with a box around the visible streak. |
| 6 | `stumps` | One box per **set** of stumps (3 stumps + bails = one box), both ends if visible. Broken stumps mid-dismissal: still one box around the cluster. |

## Box rules

- **Tight boxes**: include bat, gloves, and pads for players (they're part of the
  silhouette the model must learn), exclude shadows and reflections.
- **Occlusion**: label if ≥ 30% of the person is visible **and** you can
  identify the class without guessing. If you can only tell it's "a person",
  skip the box entirely — a wrong class label hurts more than a missing one.
- **Truncation** at frame edge: label whatever is visible, box flush with the edge.
- **Minimum size**: skip people smaller than 12 px tall (distant crowd-adjacent
  fielders in wide shots). The ball has no minimum beyond the 3×3 px rule above.
- **Never label**: crowd, ground staff, drinks carriers, players on screens/replays,
  broadcast graphics or picture-in-picture insets.

## Ambiguity tie-breaks

- Batter's runner (rare): `batsman`.
- Bowler celebrating right after the follow-through: `fielder` (the window closed).
- Keeper standing up vs back: both `wicketkeeper`.
- Two overlapping fielders: two boxes, even if heavily overlapped.
- Replay of a delivery within the broadcast: label it like live play (it's the
  same distribution); skip only picture-in-picture/split-screen frames.
- Can't decide between two classes after 10 seconds: skip the box and move on —
  note the frame in a `hard-frames.txt` list to revisit once.

## Process

1. Import `data/raw/` into Label Studio with `label_studio_config.xml`.
2. Label in passes per frame: players first, then ball, then stumps.
3. **Double-label 10%**: after finishing, re-label every 10th frame (by filename
   sort) into a second export without looking at your first pass.
4. Export both as **YOLO format** and measure agreement:

   ```bash
   python agreement.py export_a/labels export_b/labels
   ```

   Any class below **90% agreement** (IoU-matched F1): tighten its rule in this
   guide, then re-label that class everywhere before training.
