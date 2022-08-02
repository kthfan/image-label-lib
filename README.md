# image-label-lib
Transformation between yolo label, labelme label and cropped images.

## Description

| Function | Description |
| :--- | :--- |
| read_mask_from_labelme | Read polygon from labelme json file and transform into masks. |
| save_cropped_subimages_from_labelme | Read json files and images, then save the subimages in subdirectories that named by their corresponding label. Subimages are generated in sliding wnidow manner. |
| subimages_from_labelme | Read json file and image, then give the subimages and corresponding label. Subimages are generated in sliding wnidow manner. |
| labelme2yolofmt | Transform labelme format json files into yolo bounding box format txt files. |
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

3. subimages_from_labelme(json_path:str, img_path:str, window_size=(224, 224), stride_size=(112, 112)) : dict<str, np.ndarray>
```python
labeled_subimgs = save_cropped_subimages_from_labelme("./json/1.json", "./images/1.jpg")
for label, images in labeled_subimgs.items():
  print("label: ", label)
  for img in images:
    plt.imshow(img)
```

4. labelme2yolofmt(json_dir:str, output_dir:str)
```python
labelme2yolofmt("./json", "./txt")
```

5. read_yolofmt(txt_path:str) : dict<str, np.ndarray>
```python
rects_dict = read_yolofmt("./txt/1.txt")
for label, rects in rects_dict.items():
  print("label: ", label)
  for i in range(rects.shape[0]):
    print("  x: {}, y: {}, width: {}, height: {}".format(*rect[i]))
```

6. crop_by_yolofmt(txt_path:str, img_path:str, to_rect=True) : dict<str, list<np.ndarray>>
```python
sub_imgs_dict = crop_by_yolofmt("./txt/1.txt", "./images/1.jpg")
for label, sub_imgs in sub_imgs_dict.items():
  print("label: ", label)
  for img in sub_imgs:
    plt.imshow(img)
```
  
7. save_cropped_by_yolofmt(txt_dir:str, img_dir:str, output_dir:str, to_rect=True)
```python
save_cropped_by_yolofmt("./txt", "./images", "./cropped")
```

8. generate_sliding_windows(I:np.ndarray, window_size=3, stride_size=1, copy=True) : np.ndarray
```python
img = np.random.uniform(0, 1, (224, 224, 3))
sliding_windows = generate_sliding_windows(batch_images, 28, 28)
for y in range(sliding_windows.shape[0]):
  for x in range(sliding_windows.shape[1]):
    plt.subplot(sliding_windows.shape[0], sliding_windows.shape[1], y*sliding_windows.shape[1]+x+1)
    plt.imshow(sliding_windows[y, x].transpose((1, 2, 0)))
```
  


