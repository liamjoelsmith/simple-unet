from imports import *
from dataset import *
from model import *

import matplotlib.animation as animation
plt.style.use('ggplot')


"""
Reverse the diffusion process
"""
@torch.no_grad()
def predict(model):
    device = next(model.parameters()).device

    test_dataloader = load_test_dataset(batch_size=1)

    fig = plt.figure(figsize=(10,10))
    plt.axis("off")
    
    with torch.no_grad():
        for index, batch in enumerate(test_dataloader):
            image = batch
            image = image.to(device)

            ax = plt.subplot(1,2,2)
            ax.axis("off")
            plt.imshow(image[0].permute(1,2,0).detach().cpu(), cmap="gray")

            img = model(image)
            img = (img > 0.5).float()  # from memory, there is a better/more-correct way to do this

            ax = plt.subplot(1,2,1)
            ax.axis("off")
            plt.imshow(img[0].permute(1,2,0).detach().cpu(), cmap="gray", )

            break  # only want one set of images shown
            
    # plt.show()    
    plt.savefig("predict.png", transparent=True, bbox_inches='tight', pad_inches = 0)  

if __name__ == "__main__":
    model = torch.load("UNET_model_10")
    predict(model)