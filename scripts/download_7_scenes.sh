#!/bin/bash
scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')
for scene in "${scenes[@]}"; do
    url="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/${scene}.zip"
    wget "$url" -P datasets/7-Scenes
    unzip "datasets/7-Scenes/${scene}.zip" -d datasets/7-Scenes
    find "datasets/7-Scenes/${scene}" -name "*.zip" | while read filename; do
        unzip -o -d "$(dirname "$filename")" "$filename"
    done
done
cp datasets/7_scenes.json datasets/7-Scenes/camera_parameters.json