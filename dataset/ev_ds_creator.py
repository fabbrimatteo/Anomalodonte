import cv2
import imgaug.augmenters as iaa
from path import Path


NAS_PATH = Path('/nas/softechict-nas-3')
IN_PATH = NAS_PATH / 'matteo/Datasets/Spal/cables_6mm_rect/test'

OUT_PATH = NAS_PATH / 'matteo/Datasets/Spal/cables_6mm_rect/invalids'
OUT_PATH.mkdir_p()

IN_IMG_SHAPE_HW = 2048, 2448


def main(resized_h=256, resized_w=256,
         crop_x_min=812, crop_y_min=660,
         crop_side=315):
    dh = IN_IMG_SHAPE_HW[0] - (crop_y_min + crop_side)
    dw = IN_IMG_SHAPE_HW[1] - (crop_x_min + crop_side)
    displ_px = 128

    croop_good_aug = iaa.Sequential([

        iaa.Crop(px=(
            (crop_y_min, crop_y_min),
            (dw, dw),
            (dh, dh),
            (crop_x_min, crop_x_min)
        ), keep_size=False),

        iaa.Resize(size=(resized_h, resized_w))
    ])

    move_aug = iaa.Sequential([
        iaa.PerspectiveTransform(),

        iaa.Crop(px=(
            (crop_y_min - displ_px, crop_y_min + displ_px),
            (dw - displ_px, dw + displ_px),
            (dh - displ_px, dh + displ_px),
            (crop_x_min - displ_px, crop_x_min + displ_px)
        ), keep_size=False),

        iaa.CropToSquare(),
        iaa.Resize(size=(resized_h, resized_w))
    ])

    dark_aug = iaa.Sequential([
        croop_good_aug,
        iaa.Multiply(mul=(0.75, 0.5)),
        iaa.AddToBrightness(add=(-128, -32), to_colorspace=iaa.CSPACE_HSV),
        iaa.Sometimes(0.5, then_list=iaa.AddToSaturation(value=(-125, -75))),
        iaa.Sometimes(0.25, then_list=iaa.AddToHue(value=(-16, 16))),
    ])

    blur_aug = iaa.Sequential([
        croop_good_aug,
        iaa.GaussianBlur(sigma=(7, 11)),
        iaa.Sometimes(0.5, then_list=iaa.JpegCompression()),
    ])

    light_aug = iaa.Sequential([
        croop_good_aug,
        iaa.Multiply(mul=(1.25, 1.5)),
        iaa.AddToBrightness(add=(32, 128), to_colorspace=iaa.CSPACE_HSV),
        iaa.Sometimes(0.5, then_list=iaa.AddToSaturation(value=(125, 75))),
        iaa.Sometimes(0.25, then_list=iaa.AddToHue(value=(-16, 16))),
    ])

    aug_seq = iaa.OneOf([
        move_aug,
        light_aug,
        dark_aug
    ])

    for img_path in IN_PATH.files():
        img = cv2.imread(img_path)
        img_out = aug_seq.augment_image(img)

        out_name = f'invalid_{img_path.basename()}'
        print(f'$> saving {OUT_PATH / out_name}')
        cv2.imwrite(OUT_PATH / out_name, img_out)


if __name__ == '__main__':
    main()
