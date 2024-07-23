# Datasets
In this work, we used a variety of datasets, including VGGSound, AudioSet, MUSIC and AVSBench.
We used datasets of mp4 files encoded with a constant key frame rate of 16 and 360p resolution (length of short-side).
Saving frequent key-frames allows faster decoding, thus accelerating training. 

## VGGSounds
([Download](https://www.robots.ox.ac.uk/~vgg/data/vggsound/))
Expected folder structure
```
${ROOT}
${ROOT}/annotations
${ROOT}/annotations/vggsound.csv

${ROOT}/clips
${ROOT}/clips/${CLASS}
${ROOT}/clips/${CLASS}/${FILENAME}.mp4
```

## AudioSet
([Download](https://research.google.com/audioset/index.html))
Expected folder structure
```
${ROOT}
${ROOT}/annotations
${ROOT}/annotations/class_labels_indices.csv
${ROOT}/annotations/unbalanced_train_segments.csv
${ROOT}/annotations/balanced_train_segments.csv
${ROOT}/annotations/eval_segments.csv

${ROOT}/clips
${ROOT}/clips/${VID[:2]}
${ROOT}/clips/${VID[:2]}/${FILENAME}.mp4
```

## MUSIC
([Download](https://github.com/roudimit/MUSIC_dataset))
Expected folder structure
```
${ROOT}
${ROOT}/anno
${ROOT}/anno/music_solo.csv

${ROOT}/clips
${ROOT}/clips/${CLASS}
${ROOT}/clips/${CLASS}/${FILENAME}.mp4
```

## AVSBench
([Download](https://opennlplab.github.io/AVSBench/))
Expected folder structure
```
${ROOT}
${ROOT}/metadata.csv
${ROOT}/label2idx.json

${ROOT}/v1m
${ROOT}/v1m/${FILE}
${ROOT}/v1m/${FILE}/audio.wav
${ROOT}/v1m/${FILE}/frames
${ROOT}/v1m/${FILE}/labels_rgb
${ROOT}/v1m/${FILE}/labels_semantic

${ROOT}/v1s
${ROOT}/v1s/${FILE}
${ROOT}/v1s/${FILE}/audio.wav
${ROOT}/v1s/${FILE}/frames
${ROOT}/v1s/${FILE}/labels_rgb
${ROOT}/v1s/${FILE}/labels_semantic

${ROOT}/v2
${ROOT}/v2/${FILE}
${ROOT}/v2/${FILE}/audio.wav
${ROOT}/v2/${FILE}/frames
${ROOT}/v2/${FILE}/labels_rgb
${ROOT}/v2/${FILE}/labels_semantic
```