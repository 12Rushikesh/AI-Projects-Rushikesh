from pathlib import Path

def auto_accept_save(image_path, yolo_boxes, classes, W, H, dataset_img, dataset_lbl):
    dataset_img.joinpath(Path(image_path).name).write_bytes(
        open(image_path, "rb").read()
    )

    label_file = dataset_lbl / f"{Path(image_path).stem}.txt"
    with open(label_file, "w") as f:
        for b in yolo_boxes:
            if b["label"] not in classes:
                continue
            cls = classes.index(b["label"])
            x1,y1,x2,y2 = b["bbox"]
            xc = ((x1+x2)/2)/W
            yc = ((y1+y2)/2)/H
            bw = (x2-x1)/W
            bh = (y2-y1)/H
            f.write(f"{cls} {xc} {yc} {bw} {bh}\n")
