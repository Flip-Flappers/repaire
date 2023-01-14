import torch
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

model = torch.load("../net_T/pre/resnet20_check_point.pth")
model.eval()
cam_extractor = GradCAM(model)
# Get your input
for i in range(50):
    img = read_image("./A" + str(i + 1) + ".png")
    # Preprocess it for your chosen model
    tmp = to_pil_image(img)
    plt.figure()
    img2 = to_pil_image(read_image("../division/ans.png"))
    plt.subplot(1,3,1)
    plt.imshow(tmp)

    plt.subplot(1, 3, 3)
    plt.imshow(img2)
    input_tensor = normalize(img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    print(out)
    for j in range(10):
        activation_map = cam_extractor(j, out)
        # Resize the CAM and overlay it

        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # Display it
        plt.subplot(1, 3,  2)
        plt.imshow(result);
        plt.show()