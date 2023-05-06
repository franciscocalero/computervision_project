# %%
import pickle
import matplotlib.pyplot as plt

with open("../validation_data/validation_info_10.pickle", "rb") as f:
    image_list = pickle.load(f)

# %%
fig, axes = plt.subplots(nrows=len(image_list), ncols=5, figsize=(10, 4))

for validation_test in range(len(image_list)):
    axes[validation_test, 0].imshow(image_list[validation_test]['validation_image'])
    axes[validation_test, 0].set_title('Condition Image')
    axes[validation_test, 0].spines['right'].set_visible(False)
    axes[validation_test, 0].spines['top'].set_visible(False)
    axes[validation_test, 0].set_xticklabels([])
    axes[validation_test, 0].set_yticklabels([])
    axes[validation_test, 0].set_xlabel('')

    axes[validation_test, 1].imshow(image_list[validation_test]['validation_image'])
    axes[validation_test, 1].set_title('Condition Image')
    axes[validation_test, 1].spines['right'].set_visible(False)
    axes[validation_test, 1].spines['top'].set_visible(False)
    axes[validation_test, 1].set_xticklabels([])
    axes[validation_test, 1].set_yticklabels([])
    axes[validation_test, 1].set_xlabel('')

    for i in range(1, 5):
        axes[validation_test, i].imshow(image_list[validation_test]['images'][i - 1])
        axes[validation_test, i].set_title('Validation \n Image {}'.format(i))
        axes[validation_test, i].spines['right'].set_visible(False)
        axes[validation_test, i].spines['top'].set_visible(False)
        axes[validation_test, i].set_xticklabels([])
        axes[validation_test, i].set_yticklabels([])
        axes[validation_test, i].set_xlabel(image_list[validation_test]['validation_prompt'], fontsize=8)
        axes[validation_test, i].set_xlabel('')
        if i == 2:
            axes[validation_test, i].set_xlabel(image_list[validation_test]['validation_prompt'], fontsize=8)

fig.tight_layout()
plt.show()
