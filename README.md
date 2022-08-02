# image-label-lib
Transformation between yolo label, labelme label and cropped images.

## Description

| Function | Description |
| :--- | :--- |
| read_mask_from_labelme | Read polygon from labelme json file and transform into masks. |
| save_cropped_subimages_from_labelme | Read json files and images, then save the subimages in subdirectories that named by their corresponding label. Subimages are generated in sliding wnidow manner. |
| subimages_from_labelme | Read json file and image, then give the subimages and corresponding label. Subimages are generated in sliding wnidow manner. |
| labelme2yolofmt | Transform labelme format into yolo bounding box format. |
| read_yolofmt | Read bounding box in yolo format .txt file. |
| crop_by_yolofmt | Crop image into subimages by bounding boxes. |
| save_cropped_by_yolofmt | Read yolo bounding boxes .txt and images, then save cropped subimages. |
| generate_sliding_windows | Generate sliding windows. |

## Usage of functions
1.  read_mask_from_labelme(path:str) : (list<np.ndarray>, list<str>)\
```python
masks, labels = read_mask_from_labelme("./json/1.json")

for i in range(len(labels)):
  plt.subplot(len(labels), 1, i+1)
  plt.title(labels[i])
  plt.imshow(masks[i])
```

2. save_cropped_subimages_from_labelme(json_dir:str, img_dir:str, output_dir:str, window_size=(224, 224), stride_size=(112, 112))
```python
save_cropped_subimages_from_labelme("./json", "./images", "/labeled crop")
```

3. subimages_from_labelme
4.
5.

