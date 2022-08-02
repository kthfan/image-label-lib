# image-label-lib
Transformation between yolo label, labelme label and cropped images.

# Usage

| Function | Description |
| :--- | :--- |
| read_mask_from_labelme | Read labelme .json file and transform into masks. |
| save_cropped_subimages_from_labelme |  |
| subimages_from_labelme | Read json file and image, then give the subimages and corresponding label. |
| labelme2yolofmt | Transform labelme format into yolo bounding box format. |
| read_yolofmt | Read bounding box in yolo format .txt file. |
| crop_by_yolofmt | Crop image into subimages by bounding boxes. |
| save_cropped_by_yolofmt | Read yolo bounding boxes .txt and images, then save cropped subimages. |
| generate_sliding_windows | Generate sliding windows. |
