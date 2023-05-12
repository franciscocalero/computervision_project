# %%
import pickle
import matplotlib.pyplot as plt

with open("validation_data/validation_info_2.pickle", "rb") as f:
    image_list = pickle.load(f)

# %%
fig, axes = plt.subplots(nrows=len(image_list), ncols=6, figsize=(10, 3 * len(image_list)))

for validation_test in range(len(image_list)):
    for i in range(6):
        if i == 0:
            axes[validation_test, i].imshow(image_list[validation_test]['real_image'])
            axes[validation_test, i].set_title('Real Image')
        elif i == 1:
            axes[validation_test, i].imshow(image_list[validation_test]['validation_image'])
            axes[validation_test, i].set_title('Condition Image')
        else:
            axes[validation_test, i].imshow(image_list[validation_test]['images'][i - 2])
        axes[validation_test, i].set_title('Validation \n Image {}'.format(i))
        axes[validation_test, i].axis('off')

    axes[validation_test, 0].text(
        0, -0.2,
        image_list[validation_test]['validation_prompt'],
        transform=axes[validation_test, 0].transAxes,
        size=12)

fig.tight_layout()
plt.show()
